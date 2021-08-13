import csv
from numpy import ndarray
import sqlite3 as sql
from typing import Dict, List

from tokenizer import Tokenizer, TokenizerHelper


class Video:
    def __init__(self, id: int, video_id: str, title: str, url: str,
                 duration: int, uploaded_on: int, thumbnail_filepath: str,
                 num_views: int, rating: float, tags: str):
        self.id = id
        self.youtube_id = video_id
        self.title = title
        self.url = url
        self.duration = duration
        self.uploaded_on = uploaded_on
        self.thumbnail_filepath = thumbnail_filepath
        self.num_views = num_views
        self.rating = rating
        self.tags = tags


class VideoGlossary(dict):
    def __init__(self, *args, **kwargs):
        super(VideoGlossary, self).__init__(*args, **kwargs)
        self._inverse = {}
        for key, value in self.items():
            assert isinstance(value, Video)
            self._inverse[value.youtube_id] = key

    def __setitem__(self, key, value):
        assert isinstance(value, Video)

        self._inverse[value.youtube_id] = key
        super(VideoGlossary, self).__setitem__(key, value)

    def __delitem__(self, key):
        if key not in self:
            return

        if self[key].youtube_id in self._inverse:
            del self._inverse[self[key].youtube_id]
        super(VideoGlossary, self).__delitem__(key)

    def get_video_id(self, youtube_id):
        if youtube_id in self._inverse:
            return self._inverse[youtube_id]
        else:
            return -1


class SingleWordTagsGlossary(dict):
    """
    A bidirectional dictionary. Key is tag_id (integer) and value is single word tag (str)
    """
    def __init__(self, *args, **kwargs):
        super(SingleWordTagsGlossary, self).__init__(*args, **kwargs)
        self._inverse = {}
        for key, value in self.items():
            assert isinstance(value, str)
            self._inverse[value] = key

    def __setitem__(self, key, value):
        assert isinstance(value, str)
        assert len(value.split()) == 1
        self._inverse[value] = key
        super(SingleWordTagsGlossary, self).__setitem__(key, value)

    def __delitem__(self, key):
        if key not in self:
            return

        if self[key].text in self._inverse:
            del self._inverse[self[key]]
        super(SingleWordTagsGlossary, self).__delitem__(key)

    def get_tag_id(self, tag_value: str):
        if tag_value in self._inverse:
            return self._inverse[tag_value]
        else:
            return -1


class MultiWordTagsGlossary(dict):
    """
    A bidirectional dictionary. Key is tag_id (integer) and value is multi-word tag (str)
    """
    def __init__(self, *args, **kwargs):
        super(MultiWordTagsGlossary, self).__init__(*args, **kwargs)
        self._inverse = {}
        for key, value in self.items():
            assert isinstance(value, str)
            self._inverse[value] = key

    def __setitem__(self, key, value):
        assert isinstance(value, str)
        assert len(value.split()) > 1
        self._inverse[value] = key
        super(MultiWordTagsGlossary, self).__setitem__(key, value)

    def __delitem__(self, key):
        if key not in self:
            return

        if self[key].text in self._inverse:
            del self._inverse[self[key]]
        super(MultiWordTagsGlossary, self).__delitem__(key)

    def get_tag_id(self, tag_value: str):
        if tag_value in self._inverse:
            return self._inverse[tag_value]
        else:
            return -1


class TagsDataset:
    def __init__(self):

        self.video_glossary: VideoGlossary = VideoGlossary()        # video id - video object bidirectional dictionary
        self.single_word_tag_glossary: SingleWordTagsGlossary = SingleWordTagsGlossary()    # tag id - tag bidirectional dictionary
        self.multi_word_tag_glossary: MultiWordTagsGlossary = MultiWordTagsGlossary()       # tag id - tag bidirectional dictionary
        self.tag_vector_map: Dict[int, List[ndarray]] = {}      # tag id- word_vector
        self.tag_video_map: Dict[int, List[int]] = {}       # tag id- list of video id

    def _populate_video_glossary(self, csv_data_file: str):
        current_video_id = 0

        with open(csv_data_file, newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            is_header = True
            for row in csv_reader:
                if is_header:
                    is_header = False
                    continue

                youtube_video_id = row[0]
                if not youtube_video_id:
                    continue

                video_title = row[1] if row[1] is not None else "untitled"
                video_duration = -1
                try:
                    if row[2]:
                        video_duration = int(row[2])
                except ValueError:
                   pass

                video_tags = row[3] if row[3] is not None else ""
                video_views = 0
                try:
                    if row[4]:
                        video_views = int(row[4])
                except ValueError:
                   pass

                if not video_tags:
                    continue

                if self.video_glossary.get_video_id(youtube_video_id) >= 0:
                    continue

                video_obj = Video(id=current_video_id, video_id=youtube_video_id,
                                  title=video_title, url="https://www.youtube.com/watch?v=" + youtube_video_id,
                                  duration=video_duration, uploaded_on=None, thumbnail_filepath="",
                                  num_views=video_views, rating=-1, tags=video_tags)
                current_video_id += 1
                self.video_glossary[current_video_id] = video_obj

    def _populate_tags_data(self):
        current_tag_id = 0
        for video_id in self.video_glossary:
            all_tags = self.video_glossary[video_id].tags
            word_tokenizer = Tokenizer()
            tokens = word_tokenizer.get_tokens(all_tags, lemmatize=True)

            multi_word_token_text_array = []
            multi_word_token_vector_list = []
            reset_multi_word_token = False

            for token in tokens:
                if reset_multi_word_token:
                    multi_word_token_text_array = []
                    multi_word_token_vector_list = []
                    reset_multi_word_token = False

                if token.text == ',':
                    if len(multi_word_token_text_array) > 1:
                        multi_word_token_vector = TokenizerHelper.get_average_vector(multi_word_token_vector_list)
                        multi_word_token_text = " ".join(sorted(multi_word_token_text_array))
                        existing_tag_id = self.multi_word_tag_glossary.get_tag_id(multi_word_token_text)
                        if existing_tag_id < 0:
                            current_tag_id += 1
                            self.multi_word_tag_glossary[current_tag_id] = multi_word_token_text
                            self.tag_vector_map[current_tag_id] = multi_word_token_vector
                            self.tag_video_map[current_tag_id] = [video_id]
                        else:
                            if video_id not in self.tag_video_map[existing_tag_id]:
                                self.tag_video_map[existing_tag_id].append(video_id)
                    reset_multi_word_token = True
                    continue

                else:
                    existing_tag_id = self.single_word_tag_glossary.get_tag_id(token.text)
                    if existing_tag_id < 0:
                        current_tag_id += 1
                        self.single_word_tag_glossary[current_tag_id] = token.text
                        self.tag_vector_map[current_tag_id] = token.vector
                        self.tag_video_map[current_tag_id] = [video_id]
                    else:
                        if video_id not in self.tag_video_map[existing_tag_id]:
                            self.tag_video_map[existing_tag_id].append(video_id)

                    if token.text not in multi_word_token_text_array:
                        multi_word_token_text_array.append(token.text)
                        multi_word_token_vector_list.append(token.vector)

    def load_data(self):
        self._populate_video_glossary('data/data.csv')
        self._populate_tags_data()
        pass








