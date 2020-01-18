
result_dir = "result/ml1m/"
embeddings = ["08", "16", "32", "64"]
lines = []
for emb in embeddings:
    with open(result_dir+"ml1m_e"+emb+"_l201_lr0.001/ratings_matrix", 'r') as f:
        # skip headder
        next(f)
        for line in f:
            split = line.strip().split(",")
            user_id = split.pop(0)
            item_ratings = [item_rating.split(":") for item_rating in split]
            y = int(item_ratings.pop()[0])
            item_ratings.sort(key=lambda item_rating: int(float(item_rating[1])))
            item_ratings_without_rating = [int(item_rating[0]) for item_rating in item_ratings]
            ratings_as_strings = ','.join(str(val) for val in item_ratings_without_rating)
            lines.append(
                {
                    "user_id": user_id,
                    "y": y,
                    "ratings": ratings_as_strings
                }
            )

    with open(result_dir+"ml1m_"+emb+".txt", 'w') as f:
        for line in lines:
            f.write("({},{}),{}\n".format(line["user_id"], line["y"], line["ratings"]))
