from flask import Flask, render_template, request, redirect, flash, session, Response
from flask_wtf import Form
from wtforms import validators, RadioField
import numexpr as ne
import numpy as np
from equadratures import Parameter, Basis, Poly
from copy import deepcopy
import io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pickle

MAX_N = 20
app = Flask(__name__)
app.secret_key = b'sarfghasbfkahsbdklahflkajfbn'
#session.clear()
plt.rcParams["font.family"] = "Avenir", "Arial", "Helvetica", "sans-serif"
greyCol = 220/255

# not elegant...
fields = ['N', 'lower', 'upper', 'mean', 'variance', 'expression', 'univariate', 'bivariate']

@app.route('/')
def index(result=''):
    if 'formdata' in session:
        form = session['formdata']
    else:
        form = {key:'' for key in fields}
        form['N'] = '5'
    #print(form)
    return render_template('index.html', result=result, form=form)

@app.route('/integrate', methods = ['POST'])
def integrate():
    #print('ok')
    valid_distr = ['uniform', 'gaussian']
    distr = request.form['distrn']
    if distr not in valid_distr:
        result = 'Invalid distribution!'
        flash(result)
        return redirect('/')

    session['formdata'] = deepcopy(request.form)

    valid_N = [str(i) for i in range(1, MAX_N+1)]
    N = request.form['N']

    if N not in valid_N:
        result = 'Invalid number of quadrature points!'
        flash(result)
        return redirect('/')
    else:
        N = int(N)

    lower = request.form['lower']

    if lower != '':
        try:
            lower = float(lower)
        except ValueError:
            result = 'Invalid lower bound!'
            flash(result)
            return redirect('/')

    upper = request.form['upper']
    if upper != '':
        try:
            upper = float(upper)
        except ValueError:
            result = 'Invalid upper bound!'
            flash(result)
            return redirect('/')

    mean = request.form['mean']
    if mean != '':
        try:
            mean = float(mean)
        except ValueError:
            result = 'Invalid mean!'
            flash(result)
            return redirect('/')

    variance = request.form['variance']
    if variance != '':
        try:
            variance = float(variance)
        except ValueError:
            result = 'Invalid variance!'
            flash(result)
            return redirect('/')
        if variance <= 0:
            result = 'Invalid variance!'
            flash(result)
            return redirect('/')

    # TODO: alpha, beta etc.
    dimensions = request.form['dimension_params']
    expr = request.form['expression']

    if distr == 'gaussian':
        A = mean
        B = variance
    else:
        A = None
        B = None


    if dimensions == 'bivariate':
        def f(x):
            x1 = x[0]
            x2 = x[1]
            return ne.evaluate(expr)
    elif dimensions == 'univariate':
        def f(x):
            return ne.evaluate(expr)
    else:
        print('error!')

    answer = calc_integral(dimensions, f, N, distr, lower, upper, A, B)
    if answer == 'error':
        result = 'An error occurred during the polynomial quadrature process!'
        flash(result)

    return redirect('/')

def calc_integral(case, f, N, distr, lower=None, upper=None, A=None, B=None):
    if N < 2:
        return 1.0

    order = N - 1

    try:
        # Univariate setup!
        if case == 'univariate':
            my_param = Parameter(order, distr, lower=lower, upper=upper, shape_parameter_A=A, shape_parameter_B=B)
            my_basis = Basis('univariate')
            my_poly = Poly(my_param, my_basis, method='numerical-integration')
        elif case == 'bivariate':
            my_param_1 = Parameter(order, distr, lower=lower, upper=upper, shape_parameter_A=A, shape_parameter_B=B)
            my_param_2 = Parameter(order, distr, lower=lower, upper=upper, shape_parameter_A=A, shape_parameter_B=B)
            my_basis = Basis('tensor-grid')
            my_poly = Poly([my_param_1, my_param_2], my_basis, method='numerical-integration')
        my_poly.set_model(f)

    except:
        return 'error'

    if distr == 'uniform':
        interval = upper - lower
        plot_lower = lower - 1.0 * interval
        plot_upper = upper + 1.0 * interval
        session['x_lower'] = lower
        session['x_upper'] = upper
    elif distr == 'gaussian':
        stddev = np.sqrt(B)
        mean = A
        plot_lower = mean - 3.0 * stddev
        plot_upper = mean + 3.0 * stddev
        session['x_lower'] = plot_lower
        session['x_upper'] = plot_upper

    if case == 'univariate':
        plot_test = np.linspace(session['x_lower'], session['x_upper'], 50)
        session['pdf_vals'] = pickle.dumps(my_param.get_pdf(plot_test), protocol=0)
        session['fit_vals'] = pickle.dumps(my_poly.get_polyfit(plot_test).reshape(-1), protocol=0)
        session['quad_points'] = pickle.dumps(my_poly._quadrature_points, protocol=0)
        session['quad_evals'] = pickle.dumps(my_poly.get_polyfit(my_poly._quadrature_points).reshape(-1), protocol=0)
        session['integral_value'] = float(my_poly.coefficients[0])
        flash(session['integral_value'])

    elif case == 'bivariate':
        K = 10
        N = K*K
        xi = np.linspace(float(session['x_lower']), float(session['x_upper']), K)
        yi = np.linspace(float(session['x_lower']), float(session['x_upper']), K)
        xx, yy = np.meshgrid(xi, yi)
        xxi = xx.reshape(N, 1)
        yyi = yy.reshape(N, 1)
        Xout = np.hstack([xxi, yyi])
        zz = my_poly.get_polyfit(Xout).reshape(K, K)
        session['poly_surface'] =  pickle.dumps(zz, protocol=0)
        session['poly_surface_xx'] =  pickle.dumps(xx, protocol=0)
        session['poly_surface_yy'] =  pickle.dumps(yy, protocol=0)
        session['quad_points'] = pickle.dumps(my_poly._quadrature_points, protocol=0)
        session['quad_evals'] = pickle.dumps(my_poly.get_polyfit(my_poly._quadrature_points).reshape(-1), protocol=0)
        session['integral_value'] = float(my_poly.coefficients[0])
        flash(session['integral_value'])

    session['dimensions'] = case
    print('got here!')
    return session['integral_value']
@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_d.png')
def plot_png_d():
    fig = create_figure(detailed=True)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(detailed=False):
    print('Creating figure!')
    if not('formdata' in session):
        fig = Figure()
        fig.patch.set_facecolor( (greyCol, greyCol, greyCol, 0.5) )
        axis = fig.add_subplot(1, 1, 1, facecolor="none")
        return fig
    try:
        print('Line 215')
        params = session['formdata']
        fig = Figure()
        fig.patch.set_facecolor( (greyCol, greyCol, greyCol, 0.5) )
        axis = fig.add_subplot(1, 1, 1, facecolor="none")
        print('Line 222')
        if session['dimensions'] == 'univariate':
            plot_test = np.linspace(float(session['x_lower']), float(session['x_upper']), 50)
            axis.set_xlim([   float(session['x_lower']), float(session['x_upper'])   ])
            expr = params['expression']
            def f(x):
                return ne.evaluate(expr)
            f_val = f(plot_test)
            fx_ln = axis.plot(plot_test, f_val, label='f(x)')
            legends = ['f(x)', 'Area', 'Quadrature points']
            pdf_vals = pickle.loads(session['pdf_vals'])
            fit_vals = pickle.loads(session['fit_vals'])
            ymin = np.min( np.r_[f_val, pdf_vals, fit_vals] )
            ymax = np.max( np.r_[f_val, pdf_vals, fit_vals] )
            axis.set_ylim([1.1*ymin, 1.1*ymax])
            if detailed:
                axis2 = axis.twinx()
                #axis2.set_yticks([])
                pdf_ln = axis2.plot(plot_test, pdf_vals, 'g--', alpha=0.5, label='PDF (scaled)')
                fit_ln = axis.plot(plot_test, fit_vals, 'r--', alpha=0.5, label='Poly fit')
            quad_pts = pickle.loads(session['quad_points'])
            quad_evals = pickle.loads(session['quad_evals'])
            integral_val = session['integral_value']
            x_test = np.linspace(float(session['x_lower']), float(session['x_upper']), 20)
            int_fill = axis.fill_between(x_test, 0.0*x_test, f(x_test), alpha=0.5, hatch='/', label='Area')
            axis.set_title('Integral value: %f' % integral_val)
            quad_sc = axis.scatter(quad_pts, quad_evals, label='Quadrature points')
            if detailed:
                lns = [fx_ln[0], pdf_ln[0], fit_ln[0], int_fill, quad_sc]
            else:
                lns = [fx_ln[0], int_fill, quad_sc]
            legends = [l.get_label() for l in lns]
            axis.legend(lns, legends, framealpha=0.0)
        elif session['dimensions'] == 'bivariate':
            print('reached here!!!!!')
            quad_pts = pickle.loads(session['quad_points'])
            quad_evals = pickle.loads(session['quad_evals'])
            xx = pickle.loads(session['poly_surface_xx'])
            yy = pickle.loads(session['poly_surface_yy'])
            zz = pickle.loads(session['poly_surface'])
            integral_val = session['integral_value']

            norm = matplotlib.colors.Normalize(vmin=np.min(zz),vmax=np.max(zz))
            axis.contourf(xx, yy, zz, levels=20, norm=norm)
            axis.set_title('Polynomial response surface and quadrature points. \n Integral value: '+str(np.round(integral_val, 4)))
            axis.scatter(quad_pts[:,0], quad_pts[:,1], c=quad_evals, s=100, zorder=2, edgecolor='black', norm=norm, label='Quadrature points')
            #val = axis.set_xlabel('x1')
            #val2 = axis.set_ylabel('x2')
    except:
        fig = Figure()
        fig.patch.set_facecolor( (greyCol, greyCol, greyCol, 0.5) )
        axis = fig.add_subplot(1, 1, 1, facecolor="none")
        return fig

    return fig


@app.route('/clear', methods = ['GET','POST'])
def clear():
    session.clear()
    return redirect('/')


if __name__ == "__main__":
    app.run()
