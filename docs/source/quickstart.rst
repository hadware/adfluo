.. _quickstart:

==========
Quickstart
==========

Setting up a Dataset
--------------------

Let's say that we had the following yaml file describing some data that we had for
a bunch of samples (only 4 here for the sake of simplicity).

.. literalinclude:: data/example_data.yml
    :linenos:
    :language: yaml

Each sample therefore has:

* a unique identifier (``id``)
* a corresponding audio recording pointed by a path (``audio``)
* a transcription of that recording (``text``)
* one or more audio intervals (bounded by ``start`` and ``end``)
  that describe when the speaker is talking.

Let's create a sample class whose instances correspond to entries of this
``yaml`` file. This class should provide `afluo` with to methods:

* an ``id`` property that returns a unique id for that sample (in our case, easy thing!)
* a ``get_data`` method that should provide, for an instance of a sample, some
  data corresponding to the requested ``data_name``. Note that the chosen ``data_name``
  values are purely arbitrary, and up to your own choice in practice.

.. code-block:: python

    from adfluo import Sample
    import scipy # used for loading audio

    class MySample(Sample):

        def __init__(self, data: Dict):
            self.data = data

        @property
        def id(self): # has to be overriden, and must be unique (per sample)
            return self.data["id"]

        @property
        def get_data(self, data_name: str): # returns the right data for a given input name
            if data_name == "text":
                return self.data["text"]
            elif data_name == "speech":
                return [(interval["start"], interval["end"])
                        for interval in self.data["speech"]]
            elif data_name == "audio_array":
                audio_array, rate = scipy.io.wave.read(self.data["audio"])
                return {"array": audio_array, "rate": rate}

Now that we've implemented the sample class, let's implement the ``DatasetLoader`` class. It should
return an iterable of samples, and be able to tell `adfluo` how many samples there are:

.. code-block:: python

    from adfluo import DatasetLoader
    import yaml # use to load the yaml data

    class MyDataset(DatasetLoader):

        def __init__(self, data_path: str):
            with open(data_path) as yml_file:
                self.data = yaml.load(yml_file)

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            # notice that we're lazily loading samples.
            # This could be useful when samples are too big for your memory to be loaded all at once!
            for sample_data in self.data:
                yield MySample(data=sample_data)


Now that we've set up what's needed to load your data, let's proceed to the interesting part: the
feature extraction pipelines.

Setting up pipelines
--------------------

For each sample of our dataset, and for some unknown yet strangely didactic reason,
we'd like to compute the following features:
    - the number of words
    - the number of verbs
    - the number of words per spoken second
    - the number of verbs per spoken second
    - the audio length (in seconds)
    - the spoken time / total audio time ratio

Running the extraction
----------------------

TODO

Saving the Features
-------------------

TODO
