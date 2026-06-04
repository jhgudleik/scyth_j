import os
import gc
import math
import time
import psutil
import torch
import gradio as gr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "scyth_5_cpu_int8.pth"
NEW_VECTOR_NAME = "new_query.vec"
GRAY_WEIGHT = 0.07
TOP_K = 3
USE_FBGEMM = True
COSINE_THRESHOLD = 0.975
COSINE_K = 120
#COSINE_THRESHOLD = 0.965
#COSINE_K = 45.0
os.environ["CUDA_VISIBLE_DEVICES"] = ""
COSINE_OFFSET = 0.96
COSINE_SCALE = 60

# =========================================================
# CACHE
# =========================================================

_inference_cache = {
    "model": None,
    "conf": None,
    "meta": None,
    "checkpoint": None,
    "dictionaries": None,
    "tri_to_ngrm": None,
}

vector_cache = {}


# =========================================================
# UTILS
# =========================================================

def my_print_memory_usage(stage_name):
    """Печать использования RAM."""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    timestamp = (
        datetime.now() + timedelta(hours=3)
    ).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}][{stage_name}] RAM: {ram_gb:.2f} GB")

# =========================================================
# MODEL
# =========================================================

class WikiJointModel(nn.Module):
    def __init__(self, input_dim, latent_dim, linker_dim, output_dim):
        super().__init__()

        # Автоэнкодер
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)

        # LINKER =========
        self.linker = nn.Linear(latent_dim,
            linker_dim, bias=False)

        # CATEGORY HEAD =========
        self.category_head = nn.Linear(linker_dim, output_dim, bias=True)


    def forward(self, x):

        z = self.encoder(x)

        embedding = self.linker(z)

        linker_embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        # ВАЖНО:
        # classifier теперь использует normalized embedding
#        y_logits = self.category_head(linker_embedding)
# не факт

        # CATEGORY PREDICTION
        # Получаем логиты категорий
        y_logits = self.category_head(embedding)

        x_recon = None

        # tied weights decoder

        if self.training:

            z_recon = torch.matmul(
                embedding,
                self.linker.weight
            )
            x_recon = torch.matmul(
                z_recon,
                self.encoder.weight
            )
        return x_recon, y_logits, linker_embedding


# =========================================================
# LOAD MODEL
# =========================================================

def load_quantized_model(model_name):

    if _inference_cache["model"] is not None:

        print("[*] Model loaded from cache")

        return (
            _inference_cache["model"],
            _inference_cache["conf"],
            _inference_cache["meta"],
            _inference_cache["checkpoint"]
        )

    my_print_memory_usage("START MODEL LOAD")

    backend = "fbgemm" if USE_FBGEMM else "qnnpack"

    torch.backends.quantized.engine = backend

    print(f"[*] Backend: {backend}")

    start_load = time.time()

    checkpoint = torch.load(
        model_name,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"[+] Checkpoint loaded in "
        f"{(time.time() - start_load):.2f} sec"
    )

    conf = checkpoint["config"]

    meta = checkpoint["metadata"]

    if conf.get("quantized", False):

        model = checkpoint["model"]

    else:

        model = WikiJointModel(
            conf["input_dim"],
            conf["latent_size"],
            conf["linker_size"],
            conf["num_categories"]
        )

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=False
        )

    model.eval()


    '''
    with torch.no_grad():

        dummy_input = torch.zeros(
            (1, conf["input_dim"]),
            dtype=torch.float32
        )

        _, logits, emb = model(dummy_input)

        print(
            f"[OK] Forward check: "
            f"{logits.shape} | {emb.shape}"
        )

    del dummy_input, logits, emb

    gc.collect()
    '''
    _inference_cache["model"] = model
    _inference_cache["conf"] = conf
    _inference_cache["meta"] = meta
    _inference_cache["checkpoint"] = checkpoint

    my_print_memory_usage("MODEL LOADED")

    return model, conf, meta, checkpoint


# =========================================================
# LOAD DICTIONARIES
# =========================================================

def load_dictionaries(checkpoint):

    if _inference_cache["dictionaries"] is not None:

        print("[*] Dictionaries loaded from cache")

        return (
            *_inference_cache["dictionaries"],
            _inference_cache["tri_to_ngrm"]
        )

    print("[*] Loading AGR dictionaries...")

    agrs = checkpoint["agrs"]

    status_dic = agrs.get("status_dic", {})
    pairs_list = agrs.get("pairs_list", [])
    ngrm_to_tri = agrs.get("ngrm_to_tri", {})
    id_to_text = agrs.get("id_to_text", {})
    text_to_id_aux = agrs.get("text_to_id", {})
    ngrm_id_to_text = agrs.get("ngrm_id_to_text", {})
    score_dic = agrs.get("score_dic", {})

    tri_to_ngrm = {}

    for t_id, h_id in pairs_list:

        tri_to_ngrm.setdefault(
            t_id,
            []
        ).append(h_id)

    dictionaries = (
        status_dic,
        pairs_list,
        ngrm_to_tri,
        id_to_text,
        text_to_id_aux,
        ngrm_id_to_text,
        score_dic
    )

    _inference_cache["dictionaries"] = dictionaries
    _inference_cache["tri_to_ngrm"] = tri_to_ngrm

    print("[+] AGR loaded")

    return (
        status_dic,
        pairs_list,
        ngrm_to_tri,
        id_to_text,
        text_to_id_aux,
        ngrm_id_to_text,
        score_dic,
        tri_to_ngrm
    )


# =========================================================
# EMBEDDING
# =========================================================

def generate_embedding(
    text,
    model_name=MODEL_NAME,
    output_embedding_file=NEW_VECTOR_NAME
):

    if not text or not text.strip():

        return None

    text = text.lower()

    model, conf, meta, checkpoint = load_quantized_model(
        model_name
    )

    (
        status_dic,
        _,
        ngrm_to_tri,
        _,
        text_to_id_aux,
        _,
        score_dic,
        tri_to_ngrm
    ) = load_dictionaries(checkpoint)

    print(f"[*] Text length: {len(text)}")

    input_vec = torch.zeros(
        (1, conf["input_dim"]),
        dtype=torch.float32
    )

    total_trigrams_in_text = max(
        0,
        len(text) - 2
    )

    seen_white_tris = set()

    text_trigrams_ids = []

    # =====================================================
    # TRIGRAMS
    # =====================================================

    for i in range(total_trigrams_in_text):

        tid = text_to_id_aux.get(
            text[i:i+3]
        )

        if tid is not None:

            text_trigrams_ids.append(tid)

    counts_in_text = Counter(
        text_trigrams_ids
    )

    # =====================================================
    # DIRECT
    # =====================================================

    for tid in set(text_trigrams_ids):

        if status_dic.get(tid) is True:

            if tid not in seen_white_tris:

                if tid in meta["trigram2col"]:

                    seen_white_tris.add(tid)

                    col = meta["trigram2col"][tid]

                    input_vec[0, col] = 1.0

    # =====================================================
    # AGR
    # =====================================================

    processed_input_tids = set()

    for i in range(total_trigrams_in_text):

        tri_str = text[i:i+3]

        t_id = text_to_id_aux.get(tri_str)

        if (
            t_id is not None
            and t_id in tri_to_ngrm
            and t_id not in processed_input_tids
        ):

            processed_input_tids.add(t_id)

            h_ids = tri_to_ngrm[t_id]

            if not h_ids:
                continue

            def get_ngrm_score(h_idx):

                tris_in_ngrm = ngrm_to_tri.get(
                    h_idx,
                    []
                )

                return sum(
                    counts_in_text.get(s_tid, 0)
                    for s_tid in tris_in_ngrm
                )

            best_h_id = max(
                h_ids,
                key=get_ngrm_score
            )

            for s_tri_id in ngrm_to_tri.get(
                best_h_id,
                []
            ):

                if (
                    status_dic.get(s_tri_id) is True
                    and s_tri_id not in seen_white_tris
                ):

                    if s_tri_id in meta["trigram2col"]:

                        seen_white_tris.add(s_tri_id)

                        col = meta["trigram2col"][s_tri_id]

                        raw_score = score_dic.get(
                            s_tri_id,
                            0.0
                        )

                        normalized_score = max(
                            0.0,
                            min(
                                1.0,
                                raw_score / 18.0
                            )
                        )

                        final_weight = (
                            normalized_score
                            * GRAY_WEIGHT
                        )

                        input_vec[0, col] = final_weight

    # =====================================================
    # MODEL
    # =====================================================

    with torch.no_grad():

        _, _, linker_embedding = model(input_vec)

        embedding_vec = (
            linker_embedding
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )

    # =====================================================
    # SAVE
    # =====================================================

    with open(
        output_embedding_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                [
                    f"{v:.8f}"
                    for v in embedding_vec.tolist()
                ]
            )
        )

    return embedding_vec


# =========================================================
# PRELOAD
# =========================================================

print("[*] PRELOAD MODEL")

load_quantized_model(MODEL_NAME)

print("[*] MODEL PRELOADED")

# BACKEND
# PREANALYZE

STANDARD_DIR = "standard"
NEW_VECTOR_NAME = "new_query.vec"
TOP_K = 4


# =========================================================
# UTILS
# =========================================================

def my_print_memory_usage(stage_name):
    """Печать использования RAM."""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    timestamp = (
        datetime.now() + timedelta(hours=0)
    ).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}][{stage_name}] RAM: {ram_gb:.2f} GB")


my_print_memory_usage("MODEL PRELOADED")

# =========================================================
# TOPICS
# =========================================================

def load_topics_mapping():

    topics_mapping = {}
    try:
        with open(
            "standard/topics.txt",
            "r",
            encoding="utf-8"
        ) as f:
            for line in f:

                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                topics_mapping[
                    key.strip()
                ] = value.strip().strip('"')
        print("[+] topics.txt loaded")
    except FileNotFoundError:
        print("[!] topics.txt not found")
    return topics_mapping


topics_mapping = load_topics_mapping()


# =========================================================
# STANDARD VECTORS
# =========================================================


def load_standard_vector_files():
    """Загружает список стандартных векторов из файла vectors.txt"""
    try:
        with open("standard/vectors.txt", "r", encoding="utf-8") as f:
            files = [f"standard/{line.strip()}.vec" for line in f if line.strip()]
        print(f"[+] Successfully loaded {len(files)} standard vectors from vectors.txt")
        return files
    except FileNotFoundError:
        print("[!] standard/vectors.txt not found. Using hardcoded fallback.")
        return [
            "standard/1.vec", "standard/2.vec", "standard/3.vec", "standard/4.vec",
            "standard/5.vec", "standard/6.vec", "standard/7.vec", "standard/8.vec",
            "standard/9.vec", "standard/10.vec", "standard/11.vec", "standard/12.vec",
            "standard/17.vec"
        ]

standard_vectors = load_standard_vector_files()
# =========================================================
# VECTOR LOAD
# =========================================================

def load_vector(path):
    """Загружает числовой вектор из файла."""
    if path in vector_cache:
        return vector_cache[path]

    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            vec = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        vector_cache[path] = vec
        return vec
    except Exception as e:
        print(f"Ошибка при загрузке вектора {path}: {e}")
        return None


def load_standard_vectors():
    """Загружает все образцовые векторы."""
    return [(vec_file, load_vector(vec_file)) for vec_file in standard_vectors
            if load_vector(vec_file) is not None]


# =========================================================
# SIMILARITY
# =========================================================

def rescale_cosine(similarity_cosine):
    """
    Нелинейное растягивание косинусной близости.
    Особенно хорошо работает в диапазоне 0.97–0.99.
    """
    return torch.sigmoid(
        torch.tensor(
            (similarity_cosine - COSINE_OFFSET)
            * COSINE_SCALE
        )
    ).item()
def cosine_similarity(v1, v2):

    if v1 is None or v2 is None:
        return 0.0
    if np.linalg.norm(v1) == 0:
        return 0.0
    if np.linalg.norm(v2) == 0:
        return 0.0
#    return 1 - cosine(v1, v2)
    return rescale_cosine(1 - cosine(v1, v2))

def similarity_to_probability(similarity, k=5.0, threshold=0.5):
    """Преобразует косинусное сходство в вероятность с помощью сигмоиды."""
    return 1 / (1 + math.exp(-k * (similarity - threshold)))


def rank_normalized_probability(similarity, rank):
    base_prob = similarity_to_probability(
        similarity
    )
    if similarity >= 0.985:

        return base_prob
    decay_factor = 0.90 ** max(rank - 1, 0)
    return base_prob * decay_factor


# =========================================================
# VISUAL SCORE
# =========================================================

def score_similarity(similarity):
    """Визуализация сходства в формате 📶[███████████]"""
    if similarity > 0.95:
        msg = "[███████████]"
    elif similarity > 0.80:
        msg = "[██████████░]"
    elif similarity > 0.72:
        msg = "[█████████░░]"
    elif similarity > 0.62:
        msg = "[████████░░░]"
    elif similarity > 0.52:
        msg = "[███████░░░░]"
    elif similarity > 0.42:
        msg = "[██████░░░░░]"
    elif similarity > 0.32:
        msg = "[█████░░░░░░]"
    elif similarity > 0.22:
        msg = "[████░░░░░░░]"
    elif similarity > 0.18:
        msg = "[███░░░░░░░░]"
    elif similarity > 0.14:
        msg = "[██░░░░░░░░░]"
    elif similarity > 0.07:
        msg = "[█░░░░░░░░░░]"
    else:
        msg = "[░░░░░░░░░░░]"
    return msg

def get_probability_color(prob):
    """Возвращает цвет для визуальной оценки вероятности."""
    if prob > 0.72:
        return "🟢"  # Зеленый (высокая вероятность)
    elif prob > 0.42:
        return "🟡"  # Желтый (средняя)
    else:
        return "⚪️"   # Красный (низкая)



# =========================================================
# SEARCH
# =========================================================
def find_top_k_similar_vectors(query_vector, k=TOP_K):
    """Находит топ-к ближайших векторов с учётом ранга."""
    standard_vectors_list = load_standard_vectors()
    similarities = []

    for vec_file, vec in standard_vectors_list:
        if vec is None:
            continue
        similarity = cosine_similarity(query_vector, vec)
        probability = similarity_to_probability(similarity)  # Первичная вероятность

        vec_id = os.path.splitext(
            os.path.basename(vec_file)
        )[0]

        topic = topics_mapping.get(
            vec_id,
            vec_id
        )
#        topic = topics_mapping.get(vec_file, os.path.splitext(os.path.basename(vec_file))[1])        
        
        similarities.append((topic, vec_file, similarity, probability))

    similarities.sort(key=lambda x: -x[2])  # Сортировка по сходству
    return similarities[:k]


# ANALYZE

# =========================================================
# GLOBAL BEST TOPIC
# =========================================================

last_best_topic = ""
my_print_memory_usage("ANALYZE")

# =========================================================
# ANALYZE
# =========================================================

def analyze_text(text):

    global last_best_topic

    try:

        embedding = generate_embedding(text)

        if embedding is None:

            return (
                "Ошибка генерации embedding",
                "FAIL"
            )

        top_k = find_top_k_similar_vectors(embedding)

        # =====================================================
        # SAVE BEST TOPIC
        # =====================================================

        if top_k:

            last_best_topic = top_k[0][0]

        else:

            last_best_topic = ""

        result = []

        for rank, (
            topic,
            vec_file,
            similarity,
            base_prob
        ) in enumerate(top_k, 1):

            probability = rank_normalized_probability(
                similarity,
                rank
            )

            color = get_probability_color(
                probability
            )

            visual_score = score_similarity(
                probability
            )
            '''
            result.append(
                f"{rank}. {topic} ({probability:.2%})\n"
                f"Степень сходства: "
                f"{similarity:.3f} "
                f"({base_prob:.2%})\n"
                f"По образцу: "
                f"{os.path.basename(vec_file)}\n"
                f"{color}{visual_score}\n"
                "----------------------------------------"
            )
            '''
            result.append(
                f"{rank}. {topic} ({probability:.2%})\n"
                f"   Совпадение с образцом "
                f"{os.path.basename(vec_file)}: "
                f"{similarity:.3f} "
                f"({base_prob:.2%})\n"
                f"{color}{visual_score}"
#                "----------------------------------------"
            )
        return (
            "\n\n".join(result),
            f"OK | matches: {len(top_k)}"
        )

    except Exception as e:

        return (
            f"ERROR:\n{str(e)}",
            "FAIL"
        )
my_print_memory_usage("ANALYZE READY")

# DATASET

# =========================================================
# DATASET TABLE
# =========================================================

def get_dataset_table(sort_mode):

    rows = []

    try:

        topics = load_topics_mapping()

        for vec_file in standard_vectors:

            vec_id = os.path.splitext(
                os.path.basename(vec_file)
            )[0]

            topic = topics.get(
                vec_id,
                ""
            )

            txt_file = os.path.join(
                STANDARD_DIR,
                f"{vec_id}.txt"
            )

            rows.append([
                vec_id,
                topic,
                vec_file,
                txt_file
            ])

        # =================================================
        # SORT
        # =================================================

        if sort_mode == "По теме":

            rows.sort(
                key=lambda x: (
                    x[1].lower(),
                    int(x[0])
                )
            )

        else:

            rows.sort(
                key=lambda x: int(x[0])
            )

        return rows

    except Exception as e:

        return [[
            "ERROR",
            str(e),
            "",
            ""
        ]]


# =========================================================
# SELECT ROW
# =========================================================

def select_row(evt: gr.SelectData):

    return evt.index[0]

# =========================================================
# DELETE SAMPLE
# =========================================================

def delete_sample(selected_row, table_data, sort_mode):

    global standard_vectors
    global topics_mapping
    global vector_cache

    try:

        if selected_row is None:

            return (
                table_data,
                "Не выбрана строка"
            )

        # =============================================
        # GET ROW
        # =============================================
        row = table_data.iloc[selected_row]

        vec_id = str(row["ID"])

#        row = table_data[selected_row]

#        vec_id = str(row[0])

        vec_path = os.path.join(
            STANDARD_DIR,
            f"{vec_id}.vec"
        )

        txt_path = os.path.join(
            STANDARD_DIR,
            f"{vec_id}.txt"
        )

        # =============================================
        # DELETE FILES
        # =============================================

        if os.path.exists(vec_path):

            os.remove(vec_path)

        if os.path.exists(txt_path):

            os.remove(txt_path)

        # =============================================
        # UPDATE vectors.txt
        # =============================================

        vectors_path = os.path.join(
            STANDARD_DIR,
            "vectors.txt"
        )

        if os.path.exists(vectors_path):

            with open(
                vectors_path,
                "r",
                encoding="utf-8"
            ) as f:

                ids = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            ids = [
                x for x in ids
                if x != vec_id
            ]

            with open(
                vectors_path,
                "w",
                encoding="utf-8"
            ) as f:

                for item in ids:

                    f.write(f"{item}\n")

        # =============================================
        # UPDATE topics.txt
        # =============================================

        topics_path = os.path.join(
            STANDARD_DIR,
            "topics.txt"
        )

        if os.path.exists(topics_path):

            with open(
                topics_path,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

            filtered = []

            for line in lines:

                stripped = line.strip()

                if not stripped.startswith(
                    f"{vec_id} ="
                ):

                    filtered.append(line)

            with open(
                topics_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.writelines(filtered)

        # =============================================
        # RELOAD DATABASE
        # =============================================

        vector_cache.clear()

        standard_vectors = (
            load_standard_vector_files()
        )

        topics_mapping = (
            load_topics_mapping()
        )

        # =============================================
        # REFRESH TABLE
        # =============================================

        updated_table = get_dataset_table(
            sort_mode
        )

        return (
            updated_table,
            f"Удалён образец {vec_id}"
        )

    except Exception as e:

        return (
            table_data,
            f"Ошибка удаления: {e}"
        )

# BACKEND
 
TOP_K = 3
 
# =========================================================
# SHOW ADD DIALOG
# =========================================================

def show_add_dialog():

    global last_best_topic

    return (
        gr.update(
            value=last_best_topic
        ),
        gr.update(
            visible=True
        )
    )


# =========================================================
# ADD SAMPLE
# =========================================================

def add_sample_1(text, topic_text):

    global standard_vectors
    global topics_mapping
    global vector_cache

    try:

        if not text or not text.strip():

            return (
                "Ошибка: пустой текст",
                gr.update(visible=False),
                gr.update(visible=True)
            )

        if not topic_text or not topic_text.strip():

            return (
                "Ошибка: пустая тема",
                gr.update(visible=False),
                gr.update(visible=True)
            )

        embedding = generate_embedding(text)

        if embedding is None:

            return (
                "Ошибка генерации embedding",
                gr.update(visible=False),
                gr.update(visible=True)
            )

        existing_ids = []

        for vec_file in standard_vectors:

            filename = os.path.basename(vec_file)

            stem = os.path.splitext(filename)[0]

            if stem.isdigit():

                existing_ids.append(int(stem))

        new_id = (
            max(existing_ids) + 1
            if existing_ids
            else 1
        )

        # =====================================================
        # PATHS
        # =====================================================

        new_vec_path = os.path.join(
            STANDARD_DIR,
            f"{new_id}.vec"
        )

        new_txt_path = os.path.join(
            STANDARD_DIR,
            f"{new_id}.txt"
        )

        vectors_txt_path = os.path.join(
            STANDARD_DIR,
            "vectors.txt"
        )

        topics_path = os.path.join(
            STANDARD_DIR,
            "topics.txt"
        )

        # =====================================================
        # SAVE VECTOR
        # =====================================================

        np.savetxt(
            new_vec_path,
            embedding,
            fmt="%.8f"
        )

        # =====================================================
        # SAVE TEXT
        # =====================================================

        with open(
            new_txt_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

        # =====================================================
        # UPDATE vectors.txt
        # =====================================================

        with open(
            vectors_txt_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(f"{new_id}\n")

        # =====================================================
        # UPDATE topics.txt
        # =====================================================
        '''
        with open(
            topics_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
#                f'\n{new_vec_path} = "{topic_text}"\n'
                f'{new_id} = "{topic_text}"\n'            )
        '''

        with open(
            topics_path,
            "a+",
            encoding="utf-8"
        ) as f:

            f.seek(0, os.SEEK_END)

            if f.tell() > 0:

                f.seek(f.tell() - 1)

                last_char = f.read(1)

                if last_char != "\n":

                    f.write("\n")

            f.write(
                f'{new_id} = "{topic_text}"\n'
            )
        # =====================================================
        # CLEAR CACHE
        # =====================================================

        vector_cache.clear()

        standard_vectors = (
            load_standard_vector_files()
        )

        topics_mapping = (
            load_topics_mapping()
        )

        return (
            f"Образец добавлен:\n"
            f"ID: {new_id}\n"
            f"Тема: {topic_text}\n"
            f"Вектор: {new_vec_path}\n"
            f"Текст: {new_txt_path}",
            gr.update(visible=False),
            gr.update(visible=True)
        )

    except Exception as e:

        return (
            f"Ошибка: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=True)
        )


# =========================================================
# FILE HANDLER
# =========================================================

def handle_file(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        return (
            text,
            "",
            ""
        )

    except Exception as e:

        return (
            f"Ошибка файла: {e}",
            "",
            ""
        )


# =========================================================
# GRADIO
# =========================================================
my_print_memory_usage(f"TOP K: {TOP_K}")

# =========================================================
# ADD SAMPLE
# =========================================================

def add_sample(text, topic_text):

    global standard_vectors
    global topics_mapping
    global vector_cache

    try:

        if not text or not text.strip():

            return (
                "Ошибка: пустой текст",
                gr.update(visible=False),
                gr.update(visible=True),
# return
                gr.update(visible=True),
                gr.update(visible=False)
            )

        if not topic_text or not topic_text.strip():

            return (
                "Ошибка: пустая тема",
                gr.update(visible=False),
                gr.update(visible=True),
# return
                gr.update(visible=True),
                gr.update(visible=False)
            )

        embedding = generate_embedding(text)

        if embedding is None:

            return (
                "Ошибка генерации embedding",
                gr.update(visible=False),
                gr.update(visible=True),
# return
                gr.update(visible=True),
                gr.update(visible=False)
            )

        existing_ids = []

        for vec_file in standard_vectors:

            filename = os.path.basename(vec_file)

            stem = os.path.splitext(filename)[0]

            if stem.isdigit():

                existing_ids.append(int(stem))

        new_id = (
            max(existing_ids) + 1
            if existing_ids
            else 1
        )

        # =====================================================
        # PATHS
        # =====================================================

        new_vec_path = os.path.join(
            STANDARD_DIR,
            f"{new_id}.vec"
        )

        new_txt_path = os.path.join(
            STANDARD_DIR,
            f"{new_id}.txt"
        )

        vectors_txt_path = os.path.join(
            STANDARD_DIR,
            "vectors.txt"
        )

        topics_path = os.path.join(
            STANDARD_DIR,
            "topics.txt"
        )

        # =====================================================
        # SAVE VECTOR
        # =====================================================

        np.savetxt(
            new_vec_path,
            embedding,
            fmt="%.8f"
        )

        # =====================================================
        # SAVE TEXT
        # =====================================================

        with open(
            new_txt_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

        # =====================================================
        # UPDATE vectors.txt
        # =====================================================

        with open(
            vectors_txt_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(f"{new_id}\n")

        # =====================================================
        # UPDATE topics.txt
        # =====================================================
        '''
        with open(
            topics_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
#                f'\n{new_vec_path} = "{topic_text}"\n'
                f'{new_id} = "{topic_text}"\n'            )
        '''

        with open(
            topics_path,
            "a+",
            encoding="utf-8"
        ) as f:

            f.seek(0, os.SEEK_END)

            if f.tell() > 0:

                f.seek(f.tell() - 1)

                last_char = f.read(1)

                if last_char != "\n":

                    f.write("\n")

            f.write(
                f'{new_id} = "{topic_text}"\n'
            )
        # =====================================================
        # CLEAR CACHE
        # =====================================================

        vector_cache.clear()

        standard_vectors = (
            load_standard_vector_files()
        )

        topics_mapping = (
            load_topics_mapping()
        )

        return (
            f"Образец добавлен:\n"
            f"ID: {new_id}\n"
            f"Тема: {topic_text}\n"
            f"Вектор: {new_vec_path}\n"
            f"Текст: {new_txt_path}",
            gr.update(visible=False),
            gr.update(visible=True),
# return
            gr.update(visible=True),
            gr.update(visible=False)
        )

    except Exception as e:

        return (
            f"Ошибка: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=True),
# return
            gr.update(visible=True),
            gr.update(visible=False)
        )
my_print_memory_usage("ADD SAMPLE READY")



btn_width=145
# =========================================================
# GRADIO
# =========================================================
import gradio as gr
import base64

# =========================================================
# LOGO
# =========================================================
with open("scyth.png", "rb") as image_file:
    base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
# HTML-шаблон: логотип справа, надписи слева
html_content = f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img
            src="data:image/png;base64,{base64_encoded}"
            alt="Логотип"
            style="
                width: 110px;        <!-- Максимальная ширина -->
                height: auto;        <!-- Высота регулируется автоматически -->
                border-radius: 50%;
                object-fit: contain; <!-- Сохраняет пропорции -->
                max-height: 110px;   <!-- Лимит по высоте -->
            "
        >
        <div>
            <h1 style="margin: 0; font-size: 3.3em; color: #ef600f;">SCYTH-J</h1>
            <b>
            <p style="margin: 0; font-size: 1em;">
                Smart Classification
            </p>
            <p style="margin: 0; font-size: 1em; align-self: flex-end;">
                of Your Texts in Japanese
            </p>
            </b>
        </div>
    </div>
"""

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks(title="SCYTH-J",    
    css="""
    /* Создаем класс для кнопок фиксированной ширины */
    #fixed-width-button {
        width: 260px !important;

        min-width: 260px !important;
        max-width: 260px !important;

        /* 1. Убираем внутренние отступы (это самое важное) */
        padding-top: 0 !important;
        padding-bottom: 0 !important;

        /* 2. Задаем желаемую высоту кнопки */
        height: 20px !important;
        
        /* 3. Высота строки текста должна совпадать с высотой кнопки, 
           чтобы текст был по центру вертикально */
        line-height: 20px !important;
        
        /* 4. Если делаете кнопку очень маленькой, уменьшите и шрифт */
        font-size: 14px !important;

        }

    /* 1. Скрываем стандартную иконку (SVG) внутри вашего компонента 
    #my-file-upload svg {
        display: none !important;
    }
    */
    /* Скрываем надпись "Drop File Here..." */
    #my-file-upload .svelte-12ioyct {
        display: none !important;
    }
    
    
    /* 2. Находим контейнер (обычно это класс .wrap) и ставим туда свою картинку */
    #my-file-upload .wrap .svelte-12ioyct {
        /* Ссылка на вашу иконку (может быть локальный путь или URL) */
        background-image: url('./standard/icons8-archers-bow-48.png');
        
        /* Настройки отображения картинки */
        background-size: 64px 64px;      /* Размер иконки */
        background-repeat: no-repeat;    /* Не повторять */
        background-position: center 20%; /* Позиция: по центру и с отступом сверху 20% */
        
        /* Добавляем отступ сверху, чтобы текст не налезал на картинку */
        padding-top: 90px !important; 
    }
    """
) as demo:
    with gr.Tabs():

        with gr.Tab("Рубрикатор"):
            # =====================================================
            # TOP AREA
            # =====================================================
            with gr.Row():

                # =================================================
                # LEFT
                # =================================================
                with gr.Column():
                    gr.Markdown(html_content)

                # =================================================
                # MIDLE
                # =================================================
                # =============================================
                # MAIN BUTTONS
                # =============================================

                with gr.Row(
                    visible=True
                ) as main_buttons_row:

                    add_btn = gr.Button(
                        "Добавить образец",
                        scale=1,
                        elem_classes="fixed-width-button"

                    )

                    submit_btn = gr.Button(
                        "Рубрицировать",
                        variant="primary",
                        scale=1,
                        elem_classes="fixed-width-button"
                    )

                # =============================================
                # DIALOG BUTTONS
                # =============================================

                with gr.Row(
                    visible=False
                ) as dialog_buttons_row:

                    confirm_add_btn = gr.Button(
                        "Cохранить",
                        variant="primary",
                        elem_classes="fixed-width-button"
                    )

                    cancel_add_btn = gr.Button(
                        "Отмена",
                        elem_classes="fixed-width-button"
                    )


                # =================================================
                # RIGHT
                # =================================================
                with gr.Row():

                    # =============================================
                    # FILE AREA
                    # =============================================
                    with gr.Column(
                        visible=True
                    ) as add_sample_area:
                        
                        file_input = gr.File(
                            label="Выберите TXT файл",
                            show_label=True,
                            type="filepath",
                            file_types=[".txt"],
                            elem_id="my-file-upload",  # <-- ВАЖНО: этот ID связывает код с CSS
                            height="7em"
                        )

                        '''
#                        upload_btn
                        file_input = gr.UploadButton(
                            "Загрузить файл",  # Текст на кнопке
                            file_types=[".txt"],
                            type="filepath"
                        )
                        '''
                    # =============================================
                    # DIALOG
                    # =============================================
                    with gr.Column(
                        visible=False
                    ) as sample_confirm_block:
                        gr.Markdown(
                        )
                        
                        sample_topic_input = gr.Dropdown(
                            choices=sorted(
                                list(set(topics_mapping.values()))
                            ),
                            label="Тема нового образца",
                            allow_custom_value=True
                        )


                    '''
                    with gr.Row():

                        sort_mode = gr.Dropdown(
                            choices=[
                                "По имени",
                                "По теме"
                            ],
                            value="По имени",
                            label="Сортировка образцов"
                        )
                    '''
            # =====================================================
            # TEXT AREA
            # =====================================================
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Новый текст",
                        lines=10,
                        placeholder="Введите японский текст..."
                    )
                output_results = gr.Textbox(
                    label="Рубрики",
                    lines=10,
                    interactive=False
                )

            # =====================================================
            # STATUS
            # =====================================================
            status = gr.Textbox(label="Статус", interactive=False)

            # =====================================================
            # FILE HANDLER
            # =====================================================
            file_input.change(
                fn=handle_file,
                inputs=file_input,
                outputs=[
                    input_text,
                    output_results,
                    status
                ]
            )

            # =====================================================
            # ANALYZE
            # =====================================================
            submit_btn.click(
                fn=analyze_text,
                inputs=input_text,
                outputs=[
                    output_results,
                    status
                ]
            )
            # =====================================================
            # SHOW ADD DIALOG
            # =====================================================

            add_btn.click(
                fn=lambda: (
                    gr.update(value=last_best_topic),
                    gr.update(visible=True),
                    gr.update(visible=False),
# -------- 2 btns extra --------
                    gr.update(visible=False),
                    gr.update(visible=True)

                ),
                outputs=[
                    sample_topic_input,
                    sample_confirm_block,
                    add_sample_area,
# -------- 2 btns extra --------
                    main_buttons_row,
                    dialog_buttons_row
                ]
            )

            # =====================================================
            # CONFIRM ADD
            # =====================================================
            confirm_add_btn.click(
                fn=add_sample,
                inputs=[
                    input_text,
                    sample_topic_input
                ],
                outputs=[
                    status,
                    sample_confirm_block,
                    add_sample_area,
# -------- 2 btns extra --------
                    main_buttons_row,
                    dialog_buttons_row
                ]
            )

            # =====================================================
            # CANCEL
            # =====================================================
            cancel_add_btn.click(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(visible=True),
# -------- 2 btns extra --------
                    gr.update(visible=True),
                    gr.update(visible=False)

                ),
                outputs=[
                    sample_confirm_block,
                    add_sample_area,
# -------- 2 btns extra --------
                    main_buttons_row,
                    dialog_buttons_row
                ]
            )            
        
# =================================================
# DATASET TAB
# =================================================
        with gr.Tab("Образцы"):
            # =====================================================
            # TOP AREA
            # =====================================================
            with gr.Row():

                # =================================================
                # LEFT
                # =================================================

                with gr.Column():
                    gr.Markdown(html_content)

                # =================================================
                # MIDLE
                # =================================================
                with gr.Row():
                    refresh_btn = gr.Button("Обновить образцы",                    
                        scale=1,
                        elem_classes="fixed-width-button"

                    )
                    delete_btn = gr.Button(
                        "Удалить выбранный",
                        variant="primary",
                        scale=1,
                        elem_classes="fixed-width-button"

                    )

                # =================================================
                # RIGHT
                # =================================================

                with gr.Column(
                
                ):

                    # =============================================
                    # SORT AREA
                    # =============================================

                    with gr.Row():

                        sort_mode = gr.Dropdown(
                            choices=[
                                "По имени",
                                "По теме"
                            ],
                            value="По имени",
                            label="Сортировка образцов",
                        )

            # =====================================================
            # TABLE AREA
            # =====================================================

            dataset_table = gr.Dataframe(
                headers=[
                    "ID",
                    "Тема",
                    "Вектор",
                    "Текст"
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str"
                ],
                interactive=False,
                row_count=5,
                col_count=(4, "fixed"),
                max_height=300

            )

            dataset_status = gr.Textbox(
                label="Статус",
                interactive=False
            )

            # =====================================================
            # STATE
            # =====================================================

            selected_row = gr.State(
                value=None
            )

            # =====================================================
            # LOAD TABLE
            # =====================================================

            demo.load(
                fn=get_dataset_table,
                inputs=sort_mode,
                outputs=dataset_table
            )

            refresh_btn.click(
                fn=get_dataset_table,
                inputs=sort_mode,
                outputs=dataset_table
            )

            sort_mode.change(
                fn=get_dataset_table,
                inputs=sort_mode,
                outputs=dataset_table
            )

            # =====================================================
            # SELECT ROW
            # =====================================================

            dataset_table.select(
                fn=select_row,
                outputs=selected_row
            )

            # =====================================================
            # DELETE
            # =====================================================

            delete_btn.click(
                fn=delete_sample,
                inputs=[
                    selected_row,
                    dataset_table,
                    sort_mode
                ],
                outputs=[
                    dataset_table,
                    dataset_status
                ]
            )

# =========================================================
# LAUNCH
# =========================================================

import gradio as gr
import socket

# =========================================================
# Поиск свободного порта
# =========================================================
def find_free_port(start_port=7860, end_port=9000):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass

    raise RuntimeError(
        f"Не удалось найти свободный порт "
        f"в диапазоне {start_port}-{end_port}"
    )

# =========================================================
# Запуск приложения на найденном порту
# =========================================================
port = find_free_port()

print(f"Используемый порт: {port}")

demo.queue()

demo.launch(
    server_name="127.0.0.1",
    server_port=port,
    favicon_path="./standard/favicon.ico",
    inbrowser=True
)
