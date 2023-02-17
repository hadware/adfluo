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

        def __len__(self): # has to be overriden
            return len(self.data)

        def __iter__(self): # has to be overriden
            # notice that we're lazily loading samples.
            # This could be useful when samples are too big for your memory to be loaded all at once!
            for sample_data in self.data:
                yield MySample(data=sample_data)


Now that we've set up what's needed to load your data, let's proceed to the interesting part: the
feature extraction pipelines.

Setting up pipelines
--------------------

For each sample of our dataset, and for some unknown yet strangely didactic reason,
we'd like to compute the following features **for each sample in our dataset**:
    - the number of words
    - the number of verbs
    - the number of nouns
    - the audio length (in seconds)
    - the audio average energy
    - the (spoken time / total audio time) ratio
    - the number of nouns per spoken second

To compute features, we'll need to build **feature extraction pipelines**, and
to build pipelines, we'll need **processors**. Processors can be functions or
a special type of class from `adfluo` that you'll need to inherit from.

Let's start with the first pipelines: words, verbs and nouns. Let's build 3
processors: one to split the text into a list of words, one to do part of speech
tagging, and one to count words.

.. code-block:: python

    from adfluo import SampleProcessor, param

    def split_text(text: str) -> List[str]:
        return text.split(" ,.?!;")

    class PartofSpeech(SampleProcessor):
        word_type: str = param()

        def process(word_list: List[str]) -> List[str]:
            # here, through some magic (for instance using spacy),
            # either only the nouns or verbs are  returned
            ...

    def count_words(word_list: List[str]) -> int:
        return len(word_list)

Then, let's create an extractor, and register the feature extraction pipelines
to it:

.. code-block:: python

    from adfluo import Extractor, Input, Feat

    extractor = Extractor()
    extractor.add_extraction(
        Input("text") >> split_text >> count_words >> Feat("word_count")
    )
    extractor.add_extraction(
        Input("text") >> split_text >> PartOfSpeech("verb") >> count_words >> Feat("verbs_count")
    )
    extractor.add_extraction(
        Input("text") >> split_text >> PartOfSpeech("noun") >> count_words >> Feat("nouns_count")
    )

Now let's proceed to audio processors used to compute the audio duration, spoken time ratio, and average energy.

.. code-block:: python

    def audio_duration(audio_data: Dict) -> float:
        return len(audio_data["array"]) / audio_data["rate"]

    def spoken_time(intervals: List[Tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)

    def average_energy(audio_data: Dict) -> float:
        return audio_data["array"] ** 2 / len(audio_data["rate"])

Now, we can create the pipelines, still using the extractor we've created earlier:

.. code-block:: python

    extractor.add_extraction(
        Input("audio_array") >> audio_duration >> Feat("audio_duration")
    )

    extractor.add_extraction(
        Input("audio_array") >> average_energy >> Feat("avg_energy")
    )

    extractor.add_extraction(
         (Input("speech") >> spoken_time) + Feat("audio_duration")
         >> lambda dur_spoken, dur_audio : dur_spoken/ dur_audio
         >> Feat("spoken_ratio")
    )


Note that we used a lambda to avoid having to create yet another function
just compute a division. We'll now finish with the number of nouns per spoken seconds,
which won't require any new processor (we'll be using a lambda again):

.. code-block:: python

    extractor.add_extraction(
            (Feat("nouns_count") + Feat("spoken_duration"))
             >> lambda nb_nouns, dur_spoken : nb_nouns/ dur_audio
             >> Feat("nouns_per_seconds")
        )

Running the extraction and saving the features
----------------------------------------------

Now that our data is properly loaded and the extraction pipeline defined,
we can run the extraction and save the computed features:

.. code-block:: python

    # we can either extract to csv
    extractor.extract_to_csv("path/to/feature/file.csv")

    # or to a pickle file
    extractor.extract_to_pickle("path/to/feature/file.pckl")

    # or to a dict
    feat_dict = extractor.extract_to_dict()

    # or to a pandas dataframe
    feat_df = extractor.extract_to_df()

