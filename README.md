<h1>regression.py</h1>
<h2>Background</h2>
<p>
  A python script which takes line segments of 20dps and classifies them to a linear, sinusoidal or cubic fit. This script makes extensive use of matplotlib and numpy.
</p>
<p>
  Multiple line segments can be provided in one CSV. Some test datasets are provided in the repo.
  Use: python lsr.py <file.csv> [--plot]<br>
  Plot flag enbles visual representation of fit using matplotlib.
</p>
<h2>Code</h2>
<p>
  To produce the best model does NOT always directly equate to having the lowest loss function. Therefore, to preserve generality I have multiplied more complex functions by larger coefficients (similar to the L2 normalisation scheme for multiple epoch regression). I did attempt to implement a cross validation system however the datasets were so small that segmenting enough data to cross validate either ersulted in woeful overfitting or a crazy output function.
</p>
<h2>Reflection</h2>
<p>
  This script is bleoted, ineficient and overfits pretty badly. It can however give rough estimates of funtions and their coefficients. I probably would try and run it on a big set of line segments though.
  To optomise I would probably use pytorch. A library engineered and optimised for regression is always going to be more efficient and accurate than my code.
</p>
