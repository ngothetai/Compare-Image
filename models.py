from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import torch


class ImageInfo(BaseModel):
    name: str
    url: str
    width: int
    height: int


class AttributeType(str, Enum):
    texture = "texture"
    shape = "shape"
    color = "color"


class CaptionAttribute(BaseModel):
    type: AttributeType
    name: str
    key: str
    title: str
    value: str


class ClassAttribute(BaseModel):
    key: str
    question: str
    answer: str


class AnnotationItem(BaseModel):
    name: str
    contents: List[ImageInfo]
    attributes: List[CaptionAttribute]


class ClassAnnotationItem(BaseModel):
    name: str
    contents: List[ImageInfo]
    attributes: List[ClassAttribute]


class TrainingConfig(BaseModel):
    batch_size: int = 1
    num_epochs: int = 5
    learning_rate: float = 5e-5
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
