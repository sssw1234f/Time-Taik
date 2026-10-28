# -*- coding: utf-8 -*-
"""
Q&A 지능형 캐싱 시스템 모듈 (Intent-Aware Semantic Matching & Cache-Aside)
SQLite 데이터베이스를 활용하여 동일/유사 질문에 대해 LLM API 중복 호출을 방지하되,
새로운 의도(사진, 모형, 장소, 원리 등)를 가진 질문은 정밀하게 구분하여 실시간 AI로 전달합니다.
"""

import os
import sqlite3
import re
import difflib

DB_PATH = os.path.join(os.path.dirname(__file__), "qa_cache.db")

# 한국어 조사 및 용언 어미 분리 정규식
JOSA_PATTERN = re.compile(r'(?:에서|에게|으로|로|이랑|보다|까지|부터|이나|나|은|는|이|가|을|를|의|에|와|과|도)$')
VERB_STEM_PATTERN = re.compile(r'(?:이셨나요|이셨|이신|셨나요|셨|시|냐|하신|하셨|했|하|할|한|되|된|될|보는|볼|보|받은|당한)?$')

# 질문의 의도를 완전히 바꾸는 차별화 속성 키워드군 (Intent Discriminators)
# 이 단어들이 사용자 질문에 포함되어 있고 DB 질문에는 없다면, 캐시를 차단하고 실시간 AI로 분기
INTENT_DISCRIMINATORS = {
    '사진', '모형', '형태', '실제', '모습', '그림', '유물', '사료', '영상', '자료',
    '누구', '어디', '장소', '위치', '크기', '무게', '원리', '제작자', '재료', '구조',
    '언제', '시간', '사망', '사상자', '피해자', '비용', '돈', '현재',
    '고문', '몇', '발', '총', '총기', '가족', '나이', '생일', '태어난', '결과'
}

STOPWORDS = {
    "알려주", "설명해", "말씀해", "들려주", "이야기", "대해", "어떤", "무엇", "궁금", "생각",
    "각각", "있나요", "있어", "있습니까", "뭐야", "뭔가요", "인가요", "어떠", "어떻"
}


def clean_korean_word(word: str) -> str:
    """한국어 단어에서 조사 및 용언 어미를 제거하여 어근(Stem) 추출"""
    if len(word) <= 1:
        return word
    if word.startswith("만들") or word.startswith("만드"):
        return "만들"
    w = JOSA_PATTERN.sub('', word)
    w = VERB_STEM_PATTERN.sub('', w)
    return w if len(w) >= 2 else word


def normalize_question(question: str) -> str:
    """
    질문 전처리 및 정규화:
    - 소문자화, 공백 정규화, 특수기호/이모지 제거, 공통 종결 어미 제거
    """
    if not question:
        return ""
    
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"^[^\w가-힣]+", "", q).strip()
    q = re.sub(r"[?!.,~^]+$", "", q).strip()
    
    patterns = [
        r"\s*(?:을|를|은|는|이|가)?\s*(?:에\s*대해\s*|대해\s*)?(?:알려주세요|알려줘요|알려줘|알려주시겠어요|말씀해주세요|말씀해줘|설명해주세요|설명해줘|이야기해주세요|이야기해줘|들려주세요|들려줘)$",
        r"\s*(?:이었나요|였나요|인가요|인가|인지|인지요|일까요|입니까|입니까요)$",
        r"\s*(?:무엇인가요|무엇인가|무엇입니까|무엇이었나요|뭐예요|뭔가요)$"
    ]
    
    while True:
        prev = q
        for p in patterns:
            q = re.sub(p, "", q).strip()
        q = re.sub(r"(?:을|를|은|는|이|가|에|의)$", "", q).strip()
        if q == prev:
            break
        
    return q


def extract_keywords(text: str) -> set:
    """텍스트에서 정규화된 핵심 의미 토큰 세트 추출"""
    text = re.sub(r"^[^\w가-힣]+", "", text)
    text = re.sub(r"[?!.,~^()]+", " ", text)
    words = re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
    result = set()
    for raw in words:
        w = clean_korean_word(raw)
        if len(w) >= 2 and w not in STOPWORDS:
            result.add(w)
    return result


def is_semantic_intent_match(q_user: str, q_db: str) -> tuple:
    """
    의도 인식 기반 정밀 유사도 매칭:
    반환: (is_match: bool, score: float)
    """
    norm_u = normalize_question(q_user)
    norm_d = normalize_question(q_db)
    
    if norm_u == norm_d:
        return True, 1.0

    k_user = extract_keywords(q_user)
    k_db = extract_keywords(q_db)

    if not k_user or not k_db:
        return False, 0.0

    intersection = k_user & k_db
    diff_user = k_user - k_db

    # 1. 의도 분별 필터: 사용자가 사진, 모형, 위치 등 DB에 없는 완전히 새로운 속성을 요구한 경우 즉시 차단
    if diff_user & INTENT_DISCRIMINATORS:
        return False, 0.0

    # 2. 자카드 유사도 및 DB 키워드 포괄율(Coverage) 계산
    jaccard = len(intersection) / len(k_user | k_db)
    db_coverage = len(intersection) / len(k_db)
    
    # 3. 엄격한 매칭 조건:
    # Jaccard가 0.50 이상이거나, DB 핵심 키워드의 70% 이상을 포함하면서 새로운 요구 단어가 1개 이하일 때만 매칭
    is_match = (jaccard >= 0.50) or (db_coverage >= 0.70 and len(diff_user) <= 1)
    score = max(jaccard, db_coverage)
    
    return is_match, score


def get_db_connection():
    """SQLite 데이터베이스 연결 반환 (WAL 모드 적용)"""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_qa_cache_db():
    """테이블 및 인덱스 초기화"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qa_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    char_name TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    target_keyword TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_char_norm 
                ON qa_cache (char_name, normalized_question);
            """)
            conn.commit()
    except Exception:
        pass


def get_cached_qa(char_name: str, question: str):
    """
    의도 인식 기반 지능형 Cache 조회:
    1단계: 정규화 완전 일치 검사 (O(1))
    2단계: 의도 분별 필터 + 의미 유사도 정밀 검사
    
    반환: (answer, target_keyword, hit_count, match_type) 또는 None
    match_type: 'exact' (완전 일치) 또는 'semantic' (유사 질문 매칭)
    """
    if not question or not char_name:
        return None

    norm_q = normalize_question(question)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # [1단계: 정규화 완전 일치 검사]
            cursor.execute("""
                SELECT id, answer, target_keyword, hit_count
                FROM qa_cache
                WHERE char_name = ? AND normalized_question = ?
                ORDER BY id DESC
                LIMIT 1;
            """, (char_name, norm_q))
            row = cursor.fetchone()
            if row:
                row_id, answer, target_keyword, hit_count = row
                new_hit_count = (hit_count or 1) + 1
                cursor.execute("UPDATE qa_cache SET hit_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (new_hit_count, row_id))
                conn.commit()
                return answer, target_keyword, new_hit_count, "exact"

            # [2단계: 의도 분별 필터 + 의미 유사도 분석]
            cursor.execute("""
                SELECT id, original_question, answer, target_keyword, hit_count
                FROM qa_cache
                WHERE char_name = ?;
            """, (char_name,))
            candidates = cursor.fetchall()

            best_match = None
            best_score = 0.0

            for cand_id, cand_orig, cand_ans, cand_kw, cand_hits in candidates:
                is_match, score = is_semantic_intent_match(question, cand_orig)
                if is_match and score > best_score:
                    best_score = score
                    best_match = (cand_id, cand_ans, cand_kw, cand_hits)

            if best_match:
                cand_id, answer, target_keyword, hit_count = best_match
                new_hit_count = (hit_count or 1) + 1
                cursor.execute("UPDATE qa_cache SET hit_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (new_hit_count, cand_id))
                conn.commit()
                return answer, target_keyword, new_hit_count, "semantic"

    except Exception:
        return None

    return None


def save_cached_qa(char_name: str, original_question: str, answer: str, target_keyword: str = None) -> bool:
    """Cache 저장: 새로운 질문-답변 쌍을 DB에 영속화"""
    norm_q = normalize_question(original_question)
    if not norm_q or not char_name or not answer:
        return False

    if "할당량" in answer or "일시적인 통신 오류" in answer:
        return False

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, hit_count FROM qa_cache
                WHERE char_name = ? AND normalized_question = ?
                LIMIT 1;
            """, (char_name, norm_q))
            existing = cursor.fetchone()
            
            if existing:
                row_id, hit_count = existing
                cursor.execute("""
                    UPDATE qa_cache
                    SET answer = ?, target_keyword = ?, hit_count = hit_count + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                """, (answer.strip(), target_keyword or "", row_id))
            else:
                cursor.execute("""
                    INSERT INTO qa_cache (
                        char_name, normalized_question, original_question, answer, target_keyword, hit_count
                    ) VALUES (?, ?, ?, ?, ?, 1);
                """, (char_name, norm_q, original_question.strip(), answer.strip(), target_keyword or ""))
                
            conn.commit()
            return True
    except Exception:
        return False


def get_cache_stats():
    """캐시 통계 반환"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM qa_cache;")
            row = cursor.fetchone()
            total_items = row[0] if row else 0
            total_hits = row[1] if row else 0
            saved_calls = max(0, total_hits - total_items)
            return {
                "total_items": total_items,
                "total_hits": total_hits,
                "saved_calls": saved_calls
            }
    except Exception:
        return {"total_items": 0, "total_hits": 0, "saved_calls": 0}


def clean_and_seed_default_presets():
    """데이터베이스 중복 정돈 및 9대 핵심 프리셋 Q&A 고품질 시딩"""
    presets_data = [
        (
            "유관순",
            "📢 아우내 장터 만세 운동 당시의 상황을 들려주세요.",
            "1919년 4월 1일 아우내 장터 만세 운동은 3천여 명의 군중이 함께 모여 조국의 자주독립을 목놓아 외쳤던 거룩한 날이었어요. 일제 헌병의 총칼 앞에서도 물러서지 않고 태극기를 나누며 끝까지 만세를 외쳤답니다. 비록 몸은 갇혔을지라도 조국을 향한 뜨거운 정신은 결코 꺾이지 않았어요.",
            "탑골공원"
        ),
        (
            "유관순",
            "🇰🇷 직접 태극기를 만들어 배포하셨을 때 어떤 심정이셨나요?",
            "휴교령이 내려진 후 고향 천안으로 내려가 동포들의 손에 쥐여줄 태극기를 한 땀 한 땀 직접 그렸어요. 밤을 새워 태극기를 만들며 이것이 우리 민족의 자유와 존엄을 밝혀줄 횃불이 되리라 굳게 믿었습니다.",
            "태극기"
        ),
        (
            "유관순",
            "⛓️ 서대문형무소 옥중에서도 만세를 멈추지 않으신 이유는 무엇인가요?",
            "나라를 잃은 백성에게 자유와 독립보다 소중한 것은 없기 때문이에요. 옥중에서도 3·1 운동 1주년을 맞아 동지들과 함께 감옥이 울리도록 대한독립 만세를 외쳤습니다. 육체는 구속할 수 있어도 자주독립의 숭고한 의지는 결코 가둘 수 없답니다.",
            "서대문형무소"
        ),
        (
            "안중근",
            "🎯 하얼빈 역에서 이토 히로부미를 처단하신 이유가 무엇입니까?",
            "하얼빈 역의 총성은 결코 개인적인 원한에 의한 살상이 아니오. 본관은 대한국 의군 참모중장의 자격으로, 조국의 국권을 찬탈하고 동양 3국의 평화를 짓밟은 침략의 원흉 이토 히로부미의 15개 조목 죄상을 준엄하게 단죄한 것이오.",
            "하얼빈역"
        ),
        (
            "안중근",
            "🤝 단지동맹(손가락 절단) 결성 당시의 결의를 말씀해 주세요.",
            "1909년 봄 연해주에서 11명의 우국지사들과 함께 왼손 넷째 손가락 첫마디를 자르고 피로써 태극기 위에 '대한독립' 네 글자를 새겼소. 조국의 독립과 동양 평화를 위해 목숨을 바치겠다는 죽음을 넘어선 맹세였소.",
            "단지동맹"
        ),
        (
            "안중근",
            "🕊️ 옥중에서 집필하신 《동양평화론》의 핵심 정신은 무엇인가요?",
            "뤼순 감옥 옥중에서 저술한 《동양평화론》의 핵심은 한·중·일 세 나라가 서로의 자주권을 철저히 존중하고 평등하게 연대하여 외세의 침략을 막고 항구적인 공동 평화를 이루자는 것이었소. 진정한 독립이 전제될 때 비로소 참된 평화가 열리는 법이오.",
            "동양평화론"
        ),
        (
            "윤봉길",
            "💣 훙커우 공원 거사를 결심하게 된 계기는 무엇이었나요?",
            "1932년 상하이는 일제의 대륙 침략으로 먹구름이 짙게 깔려 있었고, 임시정부는 큰 침체에 빠져 있었습니다. 청년으로서 조국을 위해 목숨을 바쳐 우리 민족의 살아있는 기개와 독립 의지를 세계만방에 증명하고자 김구 선생님의 한인애국단과 함께 거사를 결단했습니다!",
            "루쉰공원"
        ),
        (
            "윤봉길",
            "🍱 물통 폭탄과 도시락 폭탄은 각각 어떤 용도였습니까?",
            "상하이 훙커우 공원 거사 당일, 침략의 주범인 일제 수뇌부를 처단하기 위해 투척용 물통 폭탄을 던졌으며, 거사 직후 일제의 손에 굴욕적인 체포를 피하고 민족의 자존을 지키기 위해 자결용 도시락 폭탄을 품에 품고 있었습니다.",
            "윤봉길"
        ),
        (
            "윤봉길",
            "✉️ 떠나기 전 두 어린 아들에게 남기신 유언의 뜻을 알려주세요.",
            "사랑하는 모순과 담 두 아들에게 '너희에게 피와 뼈가 있다면 반드시 조선을 위해 용감한 투사가 되어라'라는 유언을 남겼습니다. 한 가정을 넘어 조국의 자유와 후손들의 번영을 위해 기꺼이 목숨을 바치는 것이 청년의 숭고한 도리라 믿었기 때문입니다!",
            "매헌윤봉길의사기념관"
        )
    ]

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM qa_cache;")
            for c_name, q_txt, ans, kw in presets_data:
                norm_q = normalize_question(q_txt)
                cursor.execute("""
                    INSERT INTO qa_cache (
                        char_name, normalized_question, original_question, answer, target_keyword, hit_count
                    ) VALUES (?, ?, ?, ?, ?, 1);
                """, (c_name, norm_q, q_txt, ans, kw))
            conn.commit()
    except Exception:
        pass


# 모듈 로드 시 DB 초기화 및 기본 프리셋 시딩
init_qa_cache_db()
clean_and_seed_default_presets()
