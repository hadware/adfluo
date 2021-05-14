.. _reference:

=========
Reference
=========


External API
++++++++++++

Processors and Pipeline Elements
================================

.. autoclass:: adfluo.processors.ProcessorBase
    :members:

.. autoclass:: adfluo.SampleProcessor
    :show-inheritance:
    :members: process

.. autoclass:: adfluo.F

.. autoclass:: adfluo.Input

.. autoclass:: adfluo.Feat

Data Loading Elements
=====================

.. autoclass:: adfluo.Sample
    :members: __getitem__ , id
    :no-undoc-members:
    :exclude-members: __weakref__ , __new__, __hash__

.. autoclass:: adfluo.DatasetLoader
    :members:
    :special-members:
    :exclude-members: __weakref__

Extractor
=========

.. autoclass:: adfluo.Extractor
    :members:
    :undoc-members:


Internal API
++++++++++++

Processors
==========

Storage
=======

.. autoclass:: adfluo.storage.BaseStorage
    :members: store_feat , store_sample
    :undoc-members:

.. autoclass:: adfluo.storage.JSONStorage
    :members: write
    :show-inheritance:
    :undoc-members:

.. autoclass:: adfluo.storage.PickleStorage
    :members: write
    :show-inheritance:
    :undoc-members:

.. autoclass:: adfluo.storage.CSVStorage
    :members: write
    :show-inheritance:
    :undoc-members:

.. autoclass:: adfluo.storage.DataFrameStorage
    :members: get_value
    :show-inheritance:
    :undoc-members:

.. autoclass:: adfluo.storage.PickleStoragePerFile
    :members: write, flush
    :show-inheritance:
    :undoc-members:

Pipeline
========

pass

Extraction Graph
================

pass



