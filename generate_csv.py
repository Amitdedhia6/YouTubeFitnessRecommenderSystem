import csv
import sqlite3 as sql
from typing import List

from tokenizer import Tokenizer


def generate_csv(data_files: List[str], csv_file:str):

    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        # identifying header
        header = ['video_id', 'video_title', 'video_time', 'video_tag', 'video_views']
        writer = csv.DictWriter(file, fieldnames=header)

        # writing data row-wise into the csv file
        writer.writeheader()

        for db in data_files:
            con = sql.connect(db)
            cursor = con.cursor()
            videos_data = [[video_id, title, duration, tags, num_views]
                           for video_id, title, duration, tags, num_views in
                           cursor.execute("SELECT video_id, video_title, video_time, "
                                          "video_tag, video_views FROM workout")]

            for video_item in videos_data:
                youtube_video_id = video_item[0]
                if not youtube_video_id:
                    continue

                video_title = video_item[1] if video_item[1] is not None else "untitled"
                video_duration = video_item[2] if video_item[2] is not None else -1
                video_tags = video_item[3] if video_item[3] is not None else ""
                video_views = 0
                try:
                    if video_item[4]:
                        video_views = int(video_item[4])
                except ValueError:
                    pass

                if not video_tags:
                    continue

                word_tokenizer = Tokenizer()
                word_tokens = word_tokenizer.get_lemmatized_word_tokens(video_tags)
                word_tokens = " ".join(t for t in word_tokens)

                writer.writerow({
                    'video_id': youtube_video_id,
                    'video_title': video_title,
                    'video_time': video_duration,
                    'video_tag': word_tokens,
                    'video_views': video_views
                })

            con.close()


if __name__ == "__main__":
    data_files = ["data/workout 3.12.db", "data/elderly workout 3.16.db"]
    csv_file = 'data//data.csv'
    generate_csv(data_files, csv_file)
