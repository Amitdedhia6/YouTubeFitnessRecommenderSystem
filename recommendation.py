from typing import Dict

from dataset import TagsDataset, Video
from tokenizer import Tokenizer
from tokenizer import TokenizerHelper


class RecommendationSystem:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tags_dataset = TagsDataset()
        self.video_titles_token_repo: Dict[int, object] = {}
        self.max_results = 200

        self.tags_dataset.load_data()
        self._init_tokens_for_tags()

    def _init_tokens_for_tags(self):
        for video_id in self.tags_dataset.video_glossary:
            video_title = self.tags_dataset.video_glossary[video_id].title
            doc = self.tokenizer.get_document(video_title)
            if doc:
                self.video_titles_token_repo[video_id] = doc

    def _get_video_recommendations_based_on_video_titles(self, input_docs):
        similarity_threshold = 0.7
        video_stats_list = {}
        for input_doc in input_docs:
            for video_id in self.video_titles_token_repo:
                similarity_index = input_doc.similarity(self.video_titles_token_repo[video_id])
                if similarity_index >= similarity_threshold:
                    video_obj: Video = self.tags_dataset.video_glossary[video_id]
                    video_stats = {}
                    video_stats["video_id"] = video_obj.id
                    video_stats["title"] = video_obj.title
                    video_stats["url"] = video_obj.url
                    video_stats["duration"] = video_obj.duration
                    video_stats["num_views"] = video_obj.num_views
                    video_stats["rating"] = video_obj.rating
                    video_stats["match_count"] = 1
                    video_stats["similarity"] = similarity_index
                    video_stats_list[video_obj.id] = video_stats

        self._compute_scores_1(video_stats_list)
        video_stats_list = dict(sorted(video_stats_list.items(), key=lambda item: item[1]["score"], reverse=True))
        return video_stats_list

    def _compute_scores_1(self, video_stats_dict: Dict[int, Dict]):
        max_views_count = 0
        max_match_count = 0

        for video_id in video_stats_dict:
            video_stats_item = video_stats_dict[video_id]
            if max_views_count < video_stats_item["num_views"]:
                max_views_count = video_stats_item["num_views"]

            if max_match_count < video_stats_item["match_count"]:
                max_match_count = video_stats_item["match_count"]

        for video_id in video_stats_dict:
            video_stats_item = video_stats_dict[video_id]
            score_similarity = video_stats_item["similarity"] * 80.0
            score_views = video_stats_item["num_views"] * 20.0 / max_views_count
            score_match_count = 0   # video_stats_item["match_count"] * 30.0 / max_match_count
            video_stats_item["score"] = score_similarity + score_views + score_match_count

    def _compute_scores_tags_matching(self, video_stats_dict: Dict[int, Dict]):
        return self._compute_scores_1(video_stats_dict)

    def _get_video_recommendations_based_on_single_word_tags_matching(self, input_docs):
        similarity_threshold = 0.7
        video_stats_list: Dict[int, Dict] = {}

        for input_doc in input_docs:
            for single_word_tag_id in self.tags_dataset.single_word_tag_glossary:
                video_ids_matched_for_current_token = []
                similarity_index = TokenizerHelper.get_cosine_similarity(
                    self.tags_dataset.tag_vector_map[single_word_tag_id],
                    input_doc.vector
                )
                if similarity_index >= similarity_threshold:
                    video_id_list = self.tags_dataset.tag_video_map[single_word_tag_id]
                    for video_id in video_id_list:
                        if video_id in video_ids_matched_for_current_token:
                            video_stats = video_stats_list[video_id]
                            if video_stats["similarity"] < similarity_index:
                                video_stats["similarity"] = similarity_index
                            continue
                        else:
                            video_ids_matched_for_current_token.append(video_id)

                        video_obj: Video = self.tags_dataset.video_glossary[video_id]
                        video_stats = {}
                        if video_id in video_stats_list:
                            video_stats = video_stats_list[video_id]
                            video_stats["match_count"] = video_stats["match_count"] + 1
                            # if video_stats["similarity"] < similarity_index:
                            video_stats["similarity"] = similarity_index + video_stats["similarity"]
                        else:
                            video_stats["video_id"] = video_obj.id
                            video_stats["title"] = video_obj.title
                            video_stats["url"] = video_obj.url
                            video_stats["duration"] = video_obj.duration
                            video_stats["num_views"] = video_obj.num_views
                            video_stats["rating"] = video_obj.rating
                            video_stats["match_count"] = 1
                            video_stats["similarity"] = similarity_index
                            video_stats_list[video_obj.id] = video_stats
        self._compute_scores_tags_matching(video_stats_list)
        video_stats_list = dict(sorted(video_stats_list.items(), key=lambda item: item[1]["score"], reverse=True))

        return video_stats_list

    def _get_video_recommendations_based_on_multi_word_tags_matching(self, input_docs):
        similarity_threshold = 0.7
        video_stats_list = {}
        for input_doc in input_docs:
            if len(input_doc) <= 1:
                # in this function, we want to work with multi words only
                # single words are taken care in another function
                continue

            for multi_word_tag_id in self.tags_dataset.multi_word_tag_glossary:
                video_ids_matched_for_current_token = []
                similarity_index = TokenizerHelper.get_cosine_similarity(
                    self.tags_dataset.tag_vector_map[multi_word_tag_id],
                    input_doc.vector
                )
                if similarity_index >= similarity_threshold:
                    video_id_list = self.tags_dataset.tag_video_map[multi_word_tag_id]

                    for video_id in video_id_list:
                        if video_id in video_ids_matched_for_current_token:
                            video_stats = video_stats_list[video_id]
                            if video_stats["similarity"] < similarity_index:
                                video_stats["similarity"] = similarity_index
                            continue
                        else:
                            video_ids_matched_for_current_token.append(video_id)

                        video_obj: Video = self.tags_dataset.video_glossary[video_id]
                        video_stats = {}
                        if video_id in video_stats_list:
                            video_stats = video_stats_list[video_id]
                            video_stats["match_count"] = video_stats["match_count"] + 1
                            # if video_stats["similarity"] < similarity_index:
                            video_stats["similarity"] = similarity_index + video_stats["similarity"]
                        else:
                            video_stats["video_id"] = video_obj.id
                            video_stats["title"] = video_obj.title
                            video_stats["url"] = video_obj.url
                            video_stats["duration"] = video_obj.duration
                            video_stats["num_views"] = video_obj.num_views
                            video_stats["rating"] = video_obj.rating
                            video_stats["match_count"] = 1
                            video_stats["similarity"] = similarity_index
                            video_stats_list[video_obj.id] = video_stats

        self._compute_scores_1(video_stats_list)
        video_stats_list = dict(sorted(video_stats_list.items(), key=lambda item: item[1]["score"], reverse=True))
        return video_stats_list

    def get_video_recommendations(self, input_text: str):
        input_text_list = input_text.split(",")
        final_recommendations = {}
        input_docs = []
        for keywords in input_text_list:
            input_doc = self.tokenizer.get_document(keywords)
            input_docs.append(input_doc)

        recommendation_functions = [
            self._get_video_recommendations_based_on_video_titles,
            self._get_video_recommendations_based_on_multi_word_tags_matching,
            self._get_video_recommendations_based_on_single_word_tags_matching
        ]

        for fn in recommendation_functions:
            video_recommendation_list = fn(input_docs)
            final_recommendations.update(video_recommendation_list)
            if len(final_recommendations) >= self.max_results:
                final_recommendations = {
                    k: final_recommendations[k] for k in
                    list(final_recommendations)[:self.max_results]
                }
                break

        return final_recommendations




