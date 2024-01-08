    
from typing import Any


class LabelMapping:
    age_map = {'Kid': 0, 'Teenager': 1,'Senior': 3,'20-30s': 2,'40-50s': 4}
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
        'Fear': 5
    }
    gender_map = {
        'Male':0,
        'Female':1
    }

    def __getattribute__(self, __name: str) -> Any:
        attr = getattr(self, __name, None)
        if attr is not None:
            return attr
        else:
            setattr(self, __name, {v:k for k,v in zip(getattr(self, __name.rstrip("_rev")).items())})