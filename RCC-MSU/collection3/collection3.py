"""Collection3: Russian dataset for named entity recognition"""

import os

import datasets


logger = datasets.logging.get_logger(__name__)



_HOMEPAGE = "http://labinform.ru/pub/named_entities/index.htm"

_CITATION = """\
@inproceedings{mozharova-loukachevitch-2016-two-stage-russian-ner,
  author={Mozharova, Valerie and Loukachevitch, Natalia},
  booktitle={2016 International FRUCT Conference on Intelligence, Social Media and Web (ISMW FRUCT)},
  title={Two-stage approach in Russian named entity recognition},
  year={2016},
  pages={1-6},
  doi={10.1109/FRUCT.2016.7584769}}
"""

_DESCRIPTION = """\
Collection3 is a Russian dataset for named entity recognition annotated with LOC (location), PER (person), and ORG (organization) tags.
Dataset is based on collection Persons-1000 originally containing 1000 news documents labeled only with names of persons.
Additional labels were added by Valerie Mozharova and Natalia Loukachevitch.
Conversion to the IOB2 format and splitting into train, validation and test sets was done by DeepPavlov team.
For more details see https://ieeexplore.ieee.org/document/7584769 and http://labinform.ru/pub/named_entities/index.htm
"""

_URLS = {
    "train": os.path.join("data", "train.txt.gz"),
    "dev": os.path.join("data", "valid.txt.gz"),
    "test": os.path.join("data", "test.txt.gz"),
}


class Collection3(datasets.GeneratorBasedBuilder):
    """Collection3 dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",]
                        )
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("<DOCSTART>") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }