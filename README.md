# Watson, detective runner


Description
-----------

Watson is a framework to facilitate the research and development of machine learning problems.
The main component of this framework are:

#### The Controller
Is the runner of the orchestration of a Detective, a Datasource and a Performance Analyzer.
All those component are passed at construction time.

#### The Detective
Defines an interface, pluggable into a controller, which will wrap the specific ML Model

#### The Datasource
Defines the interface for retrieving data. Each concrete implementation can adapt to a specific DS.

#### The Transformer
Defines the interface for applying transformation to the data retrieved by the datasource. Is a component of a datasource class passed through dependency injection.

#### The Performance
Define the interface of the performance analysis. The concrete class is used at the end of the Controller execution task.


Running instruction
-------------------

### Installation
```bash
$ conda create -n watson python=3.5 numpy
$ source activate watson
$ pip install -r requirements.txt
```

To see how the full workflow works, please look at the `tests/watson/test_integration.py`





