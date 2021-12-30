.. _how_to:

========
How-To
========

This section is a list of miscellaneous task-oriented tutorials for specific usages of ``adfluo``

Dealing with large datasets
+++++++++++++++++++++++++++

This tutorial is aimed relevant if you're planning on using ``adfluo`` to extract feature from large datasets and extraction pipelines that:

- have samples that require a large amount of memory to be processed
  (e.g. : very long audio recordings)
- have so many small samples that loading all of them at once might fill
  your computer's memory (e.g: a myriad of small images)
- have implemented processors that output intermediate data that might also be very large
- some or all of the above

What follows is some advice on best to use ``adfluo`` to prevent your feature computation code from
eating all of your RAM.

Make data loading as lazy as possible
-------------------------------------

If your samples contain large files (for instance, a 2h long WAV audio file),
you should probably make the sample data loading as lazy as possible. This
means that the sample instance only contains a reference to the actual data
that is to be processed:

.. code-block:: python

    from adfluo import Sample
    from scipy.io.wavefile import read as read_wav
    from pillow import read as read_jpeg

    class MyLazySample(Sample):
        """This sample class only stores the paths to the data it references,
        and loads the actual data only when asked to."""

        def __init__(self, data_path: Path):
            self.audio_path = data_path / Path("audio.wav")
            self.img_path = data_path / Path("img.jpeg")

        def __getitem__(self, data_name: str):
            if data_name == "audio":
                return read_wav(self.audio_path)
            elif data_name == "img":
                return read_jpeg(self.img_path)

 ``adfluo`` will then take care of caching what's needed and what's not as efficiently as possible.


Use sample-wise extraction order
--------------------------------

By default, to keep a cleaner output, ``adfluo``'s extraction order is *feature-wise*, meaning that
``adfluo`` will extract ``feat_A`` for **all the samples** in the dataset, then
proceed to do the same for ``feat_b`` and so on. This is made as efficient as possible (by computing
features that share some common steps first, enabling any caches to be flushed).

One way to make ``afluo`` much leaner in term of memory usage is to run the
extraction ``sample-wise``. ``afluo`` will then extract all features for ``sample_x``,
then all features for ``sample_y``, and so on. This has the slight drawback of producing
a messier output log (if verbose mode is enabled), but otherwise doesn't have any
impact on performance.

To extract sample-wise, in any of the extraction methods from an ``Extractor``
instance, set ``extraction_order`` to ``"sample"`` :

.. code-block:: python

    extractor.extract_to_pickle(my_dataset, "path/to/feature/file.pckl", extraction_order="sample")


.. warning::

    If your extraction pipelines contain ``BatchProcessors``, it's best that you
    stick to feature-wise extraction order.

Enable streaming for the output's storage
-----------------------------------------

If both the input data and the extracted features are large, it's best to use all of
the aforementioned tricks, plus this one. This only works if you're storing
your extracted features in the "1-pickle-per-file" output or the HDF5 format.

It also requires that the storage indexing is the same as the extraction order.

.. code-block:: python

    extractor.extract_to_pickle_files(my_dataset, "pickles/folder/", stream=True)
    extractor.extract_to_hdf5(my_dataset, "path/to/data.hdf", stream=True)

Disable any form of caching
---------------------------

.. warning::

    While all of the previous solutions had no impact on performance,
    this solution can definitely have an impact, especially if some costly
    computation step is shared by several pipelines.


This is a last-ditch solution that will disable any form of caching strategy
on ``adfluo``'s side. In consequence, each step defined in a feature's computation
is systematically computed for each sample, even if it shared with another feature's
pipeline. Enable it by setting the ``no_caching`` argument:

.. code-block:: python

    extractor.extract_to_pickle(my_dataset, "path/to/feature/file.pckl", no_caching=True)


Picking the right storage format
++++++++++++++++++++++++++++++++

TODO

Sharing your extraction pipelines
+++++++++++++++++++++++++++++++++

.. code-block:: python

    raise NotImplementedError()