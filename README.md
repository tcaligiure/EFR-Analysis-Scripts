The analysis_template.ipynb was written to remove the hassle and editing in previous analysis notebooks. You should be able to simply run each cell without editing (in most cases) to
generate the plots and data arrays. These plots will be saved to the same directory that the notebook is in.
The only necessary inputs are "exp_name" and the parameter inputs on the cell below, for cavity tracking purposes. However, we are developing alternate ways of tracking these parameters,
so updates may remove this feature. The other change necessary is changing the file path from '/Users/tcaligiure/Documents/Projects/ADMX/ADMX_data.hdf5' to a path for your computer. This
is the current method of cavity tracking, where the parameters will be stored in an HDF5 file on the users computer. Each test will save the parameters and Q values to this HDF5 to serve
as a master file for all tests.
IMPORTANT: For the notebook to work, the HDF5 file that is being analyzed must be in the same directory as the analysis notebook.
IMPORTANT: Both preamble.py and AxionAnalysis.py must be stored in the Anaconda site packages folder. The path should 
be similar to: '/Users/yourUser/opt/anaconda3/lib/python3.9/site-packages'
