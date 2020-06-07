__version__ = "2.3.0"
# __version__ has to be defined in the first line

from .classifiers import (
    BoxPlot,
    EqualInterval,
    FisherJenks,
    FisherJenksSampled,
    HeadTailBreaks,
    JenksCaspall,
    JenksCaspallForced,
    JenksCaspallSampled,
    MaxP,
    MaximumBreaks,
    NaturalBreaks,
    Quantiles,
    Percentiles,
    StdMean,
    UserDefined,
    load_example,
    gadf,
    KClassifiers,
    CLASSIFIERS,
)

from .pooling import Pooled
from .greedy import greedy

dispatch = {}
for classifier in CLASSIFIERS:
    dispatch[classifier.lower()] = eval(classifier)


def classify(y, method="quantiles", **kwargs):
    """
    Helper function to dispatch specified classifier


    Parameters
    ----------

    y: array (n,1)
        Values to be classified

    method: string
         Name of classification method to used

    kwargs: dict
         Optional parameters for classifier


    Returns
    -------

    classified: MapClassifier 



    """

    ml = method.lower()
    if ml in dispatch:
        return dispatch[ml](y, **kwargs)
    else:
        print(f'{method} not a known classifier.')
