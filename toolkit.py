"""Various utility functions that did not find their way into a more "specialized" package.

Many such functions are also in jupyter_notebook. However, jupyter_notebook was really intended 
to "initialize" notebooks in a homogeneous manner, and utility functions found there way in there 
not so much by design, more by laziness. A careful migration will come... progressively.

A first "category" of helper functions appear to be related to date management. These might
potentially be moved to a "calendar_tools.py" package in the future.
"""
import inflect
import numpy as np
import pandas as pd
import pandas.io.formats.style
import time
import warnings

from IPython.display import HTML

inflect_engine = inflect.engine()
plural = inflect_engine.plural

class WarningCounter(warnings.catch_warnings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warning_count = 0

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        self.warning_count = len(self.log)
        super().__exit__(*exc_info)

class struct(dict):
    """Matlab-inspired placeholder built on a dictionnary.

    An instance of this class allows accessing/setting elements of the dictionary just like we would for 
    attributes. Syntactically, this can be very convenient, especially given that pd.DataFrame also 
    allows accessing columns as attributes.
    """
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError("No such attribute: " + name)
        #return super().__getattr__(attr)
    
    def __setattr__(self, attr, value):        
        self[attr] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)        
    

def assert_unique(values):
    """Asserts that np.unique returns a sole value, and return it.""" 
    val = np.unique(values)
    assert len(val)==1, 'Expecting a unique value'
    return val[0]
        
def unique_or_none(values, assert_unique=True):
    """Returns the sole np.unique value in `values`, or None if empty.

    If np.unique returns more than one element, the default behavior is to fail an assertion test. 
    If `assert_unique` is False, then the function simply returns None.
    """
    if values.size==0:
        return None
    
    values = np.unique(values)
    if assert_unique:
        assert values.size==1, \
            "np.unique returns more than one element, the default behavior is to fail an assertion test."
        return values[0]
    return (values[0] if values.size==1 else None)
numunique = unique_or_none

def is_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type, e.g. ipython within PyCharm
    except NameError:
        return False      # Probably standard Python interpreter

def find_the_code(func):
    import inspect

    # Get the source file and line number
    source_file = inspect.getsourcefile(func)
    line_number = inspect.getsourcelines(func)[1]

    print(f'Defined in file: {source_file}, line: {line_number}')    

def nan_equal_dataframe(df1, df2):
    equal = False
    shape_equal = df1.shape == df2.shape
    columns_equal = df1.columns.equals(df2.columns)            
    if shape_equal and columns_equal:
        eq_entries = (df1 == df2)
        eq_nan = df1.isna() & df2.isna()
        eq_nan_obj = (df1.isna() & (df2=='')) | ((df1=='') & df2.isna())
        equal = np.all(eq_entries | eq_nan | eq_nan_obj)
    return equal

def nancorr(ar, a2=None):
    if a2 is not None:
        return nancorr([ar, a2])
    
    assert isinstance(ar, list), "Other classes than list not implemented yet"
    
    import numpy.ma as ma
    msk = np.ones(ar[0].shape, dtype=bool)
    for no,a_i in enumerate(ar):
        a_i = ma.masked_invalid(a_i)
        msk = msk & ~a_i.mask
        ar[no] = a_i
        
    return np.corrcoef([nn[msk] for nn in ar])

def object_vars_to_string(obj):
    string = obj.__class__.__name__ + '(\n'    
    fields = vars(obj)
    for no,name in enumerate(fields):
        if name.startswith('_'):
            continue
        string += "    %s = %r" % (name, fields[name])
        if no < len(fields)-1:
            string += ','
        string += '\n'
    string += ')'
    return string    
    
def print_versions():
    """Versions of the critical dependencies"""
    from platform import python_version
    print('Python:',python_version())
    print('Numpy:',np.__version__)
    print('Pandas:',pd.__version__)    

def printdf(df, T=True, head_tail=None):
    #import pdb; pdb.set_trace()
    
    if isinstance(df, type(np.array([]))):
        df = pd.DataFrame(df)
    
    if isinstance(df, pd.Series):
        if T:
            df = df.to_frame().transpose()
        else:
            df = df.to_frame()

    if head_tail is not None:
        from itertools import chain
        lx = df.shape[0]
        ix = chain(range(head_tail), range(lx-head_tail,lx))
        df = df.iloc[ix]

    #import pdb; pdb.set_trace()
    if not is_jupyter_notebook():
        print(df)
        return
            
    if isinstance(df, pd.io.formats.style.Styler):
        display(df)        
    else:
        display(HTML(df.to_html()))

__tic = [] # Last in, first out (LIFO)
def tic():
    """Marks the beginning of a time interval"""    
    global __tic
    __tic.append(time.perf_counter())

def toc(do_print=True):
    """Prints the time difference since the last tic (that was not toc'ed yet; LIFO)."""
    global __tic
    dt = time.perf_counter() - __tic.pop()
    if do_print:
        print("Elapsed time: %f seconds.\n"%dt)
    return dt


#### Date utils:

def subcalendar(calendar, after=None, first_date=None, last_date=None, before=None):
    """Subsample the given calendar, or dataframe based on its index.    

    If the first argument is a dataframe, its index will be assumed to be the calendar to subsample 

    Args:
        calendar: the calendar or dataframe to subsample from
        after: excluding dates before or on the `after` date (mutually exclusive with `first_date`)
        first_date: starting on first_date, included if it exists (mutually exclusive with `after`)
        last_date: ending on last_date, included if it exists (mutually exclusive with `before`)
        before: excluding dates after or on the `before` date (mutually exclusive with `last_date`)
    """
    df = None
    if isinstance(calendar, pd.DataFrame):
        df = calendar
        calendar = df.index
    if len(calendar)==0:        
        return (calendar if df is None else df)
    
    if isinstance(calendar[0],pd.Timestamp):
        timestamp = lambda date: date if not isinstance(date,str) \
                                        else pd.Timestamp(date+' 00:00:00')
    else:
        timestamp = lambda date: date if not isinstance(date,str) \
                                        else pd.Timestamp(date+' 00:00:00').date()    

    if after is not None:
        calendar = calendar[calendar > timestamp(after)]
        assert first_date is None, 'after and last_date cannot be used jointly'
    elif first_date is not None:
        calendar = calendar[calendar >= timestamp(first_date)]
        
    if last_date is not None:
        calendar = calendar[calendar <= timestamp(last_date)]
        assert before is None, 'first_date and before cannot be used jointly'            
    elif before is not None:
        calendar = calendar[calendar < timestamp(before)]

    if df is None:
        return calendar
    return df.loc[calendar]

def subsample(df, sx, pad=1):
    """Returns rows where dx is True, with the previous/next `pad` rows."""
    loc = np.array([df.index.get_loc(ix) for ix in df.index[sx]])
    xloc = np.maximum(0,np.concatenate((loc-pad, loc, loc+pad)))
    xloc = np.minimum(xloc, len(df.index)-1)
    return df.iloc[np.unique(xloc)].sort_index()
    
def datetime2yyyymmdd(dt):
    if not hasattr(dt,'year'):
        dt = pd.to_datetime(dt)
    return int(1e4*dt.year + 1e2*dt.month + dt.day)

def yyyymmdd2timestamp(yyyymmdd):
    #breakpoint()
    dd     = np.mod(yyyymmdd, 100)
    yyyymm = (yyyymmdd - dd)/100;  
    mm     = np.mod(yyyymm, 100);
    yyyy   = (yyyymm - mm)/100;
    if not hasattr(yyyymmdd,'__len__'):
        return pd.Timestamp('%d-%02d-%02d'%(yyyy,mm,dd))
    return [pd.Timestamp('%d-%02d-%02d'%(yyyy[no],mm[no],dd[no])) for no in range(len(yyyymmdd))]

def date2str(dt):
    if isinstance(dt, np.datetime64):
        dt = pd.to_datetime(dt)
    return dt.strftime('%Y/%m/%d')
    
