import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

#data retrieval from filepath using pandas
def loadData(dp):
    data = pd.read_csv(dp)
    return data

#calculate error of model
def squareError(y, y_hat):
    return np.sum((y - y_hat) ** 2)

#generate xs ys and error value of linear segment
def generateLinearY(a, b, datax):
    ys = (datax * b) + a    
    return (ys)

#generate xs ys and error value of exponential segment
def generateExpY(a, b, datax):
    ys = (np.exp(datax) * b) + a
    return (ys)

#generate xs ys and error value of exponential segment
def generateSineY(a, b, datax):
    ys = (np.sin(datax) * b) + a
    return (ys)

#generate xs ys and error value of quadratic segment
def generateQuadY(a, b, c, datax):
    ys = (datax * b) + a + ((datax ** 2) * c )
    return (ys)

#generate xs ys and error value of cubic segment
def generateCubeY(a, b, c, d, datax):
    ys = (datax * b) + a + ((datax ** 2) * c) + ((datax ** 3) * d)
    return (ys)

def plotLine(xs, ys):
    plt.plot(xs, ys, 'r-', lw=1)
    
def bestModel(datax, datay):
    linCV = 0
    expCV = 0
    quadCV = 0
    cubeCV = 0            
    #the number of samples cross validate with - larger is weaker validation
    CVC = 4
        
    #separating out cross validation data
    crossvalx = datax[::CVC]
    crossvaly = datay[::CVC]
    dataxinc = datax[CVC-1::CVC]
    datayinc = datay[CVC-1::CVC]
        
    #find regression coefficients for reduced daatasets
    linearregression = linearLeastSquares(dataxinc, datayinc)
    expregression = expLeastSquares(dataxinc, datayinc)
    quadregression = quadLeastSquares(dataxinc, datayinc)
    cuberegression = cubeLeastSquares(dataxinc, datayinc)
        
    #find XY sequences for reduced datasets
    linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
    expModel = generateExpY(expregression[0], expregression[1], datax)
    quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
    cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
        
    #find error between reduced dataset model and crossvalidation data
    linCV += squareError(crossvaly, linearModel[::CVC])
    expCV += squareError(crossvaly, expModel[::CVC])
    quadCV += squareError(crossvaly, quadModel[::CVC])
    cubeCV += squareError(crossvaly, cubeModel[::CVC])
        
    if linCV <= expCV and linCV <= cubeCV:
        return 1
    elif expCV <= cubeCV:
        return 2
    else:
        return 4

def bestModelStartEndCV(datax, datay):  
         
#the number of samples cross validate with - larger is weaker validation
    CVC = 4
        
        #separating out cross validation data
    crossvalx = np.concatenate([datax[:CVC], datax[:-CVC]])
    crossvaly = np.concatenate([datax[:CVC], datay[:-CVC]])
    dataxinc = datax[CVC + 1:datax.size - CVC]
    datayinc = datay[CVC + 1:datay.size - CVC]
        
        #find regression coefficients for reduced daatasets
    linearregression = linearLeastSquares(dataxinc, datayinc)
    expregression = expLeastSquares(dataxinc, datayinc)
    quadregression = quadLeastSquares(dataxinc, datayinc)
    cuberegression = cubeLeastSquares(dataxinc, datayinc)
        
        #find XY sequences for reduced datasets
    linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
    expModel = generateExpY(expregression[0], expregression[1], datax)
    quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
    cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
        
        #find error between reduced dataset model and crossvalidation data
    linCV = squareError(crossvaly, np.concatenate([linearModel[:CVC], linearModel[:-CVC]]))
    expCV = squareError(crossvaly, np.concatenate([expModel[:CVC], expModel[:-CVC]]))
    quadCV = squareError(crossvaly, np.concatenate([quadModel[:CVC], quadModel[:-CVC]]))
    cubeCV = squareError(crossvaly, np.concatenate([cubeModel[:CVC], cubeModel[:-CVC]]))
        
    if linCV <= expCV and linCV <= quadCV and linCV <= cubeCV:
        return 1
    elif expCV <= quadCV and expCV <= cubeCV:
        return 2
    elif quadCV <= cubeCV:
        return 3
    else:
        return 4

#cross validation on last CVC points
def bestModelEndCV(datax, datay):  

    CVC = 5   
        #separating out cross validation data
    crossvalx = datax[:-CVC]
    crossvaly = datay[:-CVC]
    dataxinc = datax[:datax.size - CVC]
    datayinc = datay[:datay.size - CVC]
        
        #find regression coefficients for reduced daatasets
    linearregression = linearLeastSquares(dataxinc, datayinc)
    expregression = expLeastSquares(dataxinc, datayinc)
    quadregression = quadLeastSquares(dataxinc, datayinc)
    cuberegression = cubeLeastSquares(dataxinc, datayinc)
        
        #find XY sequences for reduced datasets
    linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
    expModel = generateExpY(expregression[0], expregression[1], datax)
    quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
    cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
        
        #find error between reduced dataset model and crossvalidation data
    linCV = squareError(crossvaly, linearModel[:-CVC])
    expCV = squareError(crossvaly, expModel[:-CVC])
    quadCV = squareError(crossvaly, quadModel[:-CVC])
    cubeCV = squareError(crossvaly, cubeModel[:-CVC])
        
    if linCV <= expCV and linCV <= quadCV and linCV <= cubeCV:
        return 1
    elif expCV <= quadCV and expCV <= cubeCV:
        return 2
    elif quadCV <= cubeCV:
        return 3
    else:
        return 4

#use a simple weight system to find the function with the lowest error        
def bestModelLinear(datax, datay):
    #defining weights for function errors
    linWeight = 3
    sinWeight = 8
    cubeWeight = 40
       
        #find regression coefficients for reduced daatasets
    linearregression = linearLeastSquares(datax, datay)
    sinregression = sinLeastSquares(datax, datay)
    #quadregression = quadLeastSquares(datax, datay)
    cuberegression = cubeLeastSquares(datax, datay)
        
        #find XY sequences for reduced datasets
    linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
    sinModel = generateSineY(sinregression[0], sinregression[1], datax)
    #quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
    cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
        
        #find error between reduced dataset model and crossvalidation data
    linCV = squareError(datay, linearModel) * linWeight
    sinCV = squareError(datay, sinModel) * sinWeight
    #quadCV = squareError(datay, quadModel)
    cubeCV = squareError(datay, cubeModel) * cubeWeight
        
    if linCV <= sinCV and linCV <= cubeCV:
        return 1
    elif sinCV <= cubeCV:
        return 2
    else:
        return 4
    
#generate graphics plot of data and print error
def generatePlot(datax, datay):
    plt.scatter(datax, datay, linewidths = 0.5)
    #found crossvalidation functions didnt work
    model = bestModelLinear(datax, datay)
    
#    if model == 1:
#        linearregression = linearLeastSquares(datax, datay)
#        linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
#        plotLine(datax, linearModel)
#        print("linear \n")
#        return squareError(datay, linearModel)
#    elif model == 2:
#        sinregression = sinLeastSquares(datax, datay)
#        sinModel = generateExpY(sinregression[0], sinregression[1], datax)
#        plotLine(datax, sinModel)
#        print("exponential \n")
#        return squareError(datay, sinModel)
#    elif model == 3:
#        quadregression = quadLeastSquares(datax, datay)
#        quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
#        plotLine(datax, quadModel)
#        print("quadratic \n")
#        return squareError(datay, quadModel)
#    else:
#        cuberegression = cubeLeastSquares(datax, datay)
#        cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
#        plotLine(datax, cubeModel)
#        print("cubic \n")
#        return  squareError(datay, cubeModel)
    
    if model == 1:
        linearregression = linearLeastSquares(datax, datay)
        linearModel = generateLinearY(linearregression[0], linearregression[1], datax)
        plotLine(datax, linearModel)
        print("linear \n")
        return squareError(datay, linearModel)
    elif model == 2:
        sinregression = sinLeastSquares(datax, datay)
        sinModel = generateSineY(sinregression[0], sinregression[1], datax)
        plotLine(datax, sinModel)
        print("sine \n")
        return squareError(datay, sinModel)
    elif model == 3:
        quadregression = quadLeastSquares(datax, datay)
        quadModel = generateQuadY(quadregression[0], quadregression[1], quadregression[2], datax)
        plotLine(datax, quadModel)
        print("quadratic \n")
        return squareError(datay, quadModel)
    else:
        cuberegression = cubeLeastSquares(datax, datay)
        cubeModel = generateCubeY(cuberegression[0], cuberegression[1], cuberegression[2], cuberegression[3], datax)
        plotLine(datax, cubeModel)
        print("cubic \n")
        return  squareError(datay, cubeModel)

#find sine least squares coefficients
def sinLeastSquares(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

#find linear least squares coefficients 
def linearLeastSquares(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v 

#find lquadratic least squares coefficients 
def quadLeastSquares(x, y):
    ones = np.ones(x.shape)
    xsquared = np.power(x, 2)
    x_e = np.column_stack((ones, x, xsquared))
    
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v 

#find cubic least squares coefficients 
def cubeLeastSquares(x, y):
    ones = np.ones(x.shape)
    xsquared = np.power(x, 2)
    xcubed = np.power(x, 3)
    x_e = np.column_stack((ones, x, xsquared, xcubed))
    
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v 
    
 
#find exponential least squares coefficients
def expLeastSquares(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.exp(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def get_parser():
    """Returns parser object to retrieve command line argunents."""

    parser = argparse.ArgumentParser(description="Run least squares regression.")
    parser.add_argument(
            'filepath',
            type=str,
            help=".csv filepath of target data",
            )
    parser.add_argument(
            '--plot',
            action='store_true',
            help="type --plot to see plot of data and regression.",
            )
    return parser

def main():
    # Retrieve filepath from argument list
    parser = get_parser()
    args = vars(parser.parse_args())
    filepath = args.pop('filepath')
    plot = args.pop('plot')
    

    #retrieve data from provided filepath
    data = loadData(filepath)
    dataX = np.array(data)[:,0]
    dataY = np.array(data)[:,1]
    lines = data.shape[0] + 1
    #initialise sum error
    err = 0
    
    #iterate over each segment
    for i in range(0, np.int_((lines/20))):
        #generate and plot linear graph, print error squared value
        err += generatePlot(dataX[(i * 20):(i * 20) + 20], dataY[(i * 20):(i * 20) + 20])
    
    print(err, '\n')
    if plot:
        plt.show()
    
    
main()