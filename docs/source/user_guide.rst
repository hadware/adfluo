.. _user_guide:

==========
User Guide
==========


Datasets
--------

``DatasetLoader`` are a very light abstraction from ``adfluo``, largely
inspired by Pytorch's `Dataloaders <https://pytorch.org/docs/stable/data.html>`_ .

Their goal is to provide an interface by which ``adfluo`` can:

- know the number of samples in your dataset, through the ``__len__`` magic method.
- load the samples of your dataset (one-by-one), through the ``__iter___`` magic method.

Everything else is up to you.

The samples your ``DatasetLoader`` loads contain the "input" data for your pipeline. There are
two basic things to understand about samples:

- they can contain any data you want, but any data they contain should be accessible using a string identifier, through
  the ``__getitem__`` method.
- all of your dataset's samples should have a unique string identifier

Implementing your own DatasetLoader and Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we have a dataset containing audio files and a corresponding transcription. They are stored
using the following file hierarchy:

.. code-block::

    dataset/
    ├── audio
    │   ├── sample_1.flac
    │   ├── sample_2.flac
    │   └── sample_3.flac
    └── transcripts
        ├── transcript_1.txt
        ├── transcript_2.txt
        └── transcript_3.txt

For each sample, we'd like to load both the audio and its transcript, as a string.
First, let's implement our own ``Sample`` class

.. code-block:: python

    from adfluo import Sample
    import librosa

    class TranscriptedAudioSample(Sample):

        # init method can be whatever you want
        def __init__(self, audio_path: Path, transcript_path: Path):
            self.audio_path = audio_path
            with open(trancript_path) as trancript_file:
                self.transcript = trancript_file.read()

        # has to be overriden, and must return a unique id (per sample)
        @property
        def id(self) -> str:
            return self.audio_path.stem

        def __getitem__(self, data_name: str):
            if data_name == "audio":
                # we load the audio, and return an (audio_array, sample_rate) tuple
                audio_array, rate = librosa.load(self.audio_path)
            elif data_name = "transcript":
                return self.transcript


Using Python primitives as a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Processors
----------

TODO: List and explain the types of processors

Sample Processors
~~~~~~~~~~~~~~~~~

TODO

Using functions and lambdas as processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Batch processors
~~~~~~~~~~~~~~~~

TODO


Pipelines
---------

Basics
~~~~~~

TODO

Advanced Input/Output
~~~~~~~~~~~~~~~~~~~~~

TODO

Advanced Pipelines Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Extraction
----------

Storage options
~~~~~~~~~~~~~~~

TODO

Extraction options
~~~~~~~~~~~~~~~~~~~

TODO
