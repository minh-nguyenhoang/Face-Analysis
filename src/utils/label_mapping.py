    
from typing import Any


class LabelMapping:
    age_map = {'Baby':0, 'Kid': 1, 'Teenager': 2,'Senior': 4,'20-30s': 3,'40-50s': 5}
    race_map = {
        'Mongoloid': 0,
        'Caucasian': 1,
        'Negroid': 2
    }
    masked_map = {
        'masked': 0,
        'unmasked':1
    }
    skintone_map = {
        'light': 0,
        'mid-light': 1,
        'mid-dark' :2,
        'dark': 3
    }
    emotion_map = {
        'Happiness': 0,
        'Neutral':1,
        'Sadness':2,
        'Anger':3,
        'Surprise':4,
        'Fear': 5,
        'Disgust': 6
    }
    gender_map = {
        'Male':0,
        'Female':1
    }

    @staticmethod
    def get(name):
        return LabelMapping.__getattribute__(LabelMapping, name)

    def __getattribute__(self, __name: str) -> Any:
        attr = getattr(self, __name, None)
        if attr is not None:
            return attr
        else:
            setattr(LabelMapping, __name, {v:k for k,v in getattr(self, __name.rstrip("_rev")).items()})
            return self.__getattribute__(self, __name)