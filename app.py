import streamlit as st
import os
import urllib.request
import urllib.parse
import json
import re
import google.generativeai as genai

# [1. 데이터베이스] - 공공데이터를 딕셔너리로 구조화
knowledge_base = {
    "유관순": {
        "img": "yugwansun.png",
        "fact": "1919년 3월 1일 아우내 장터 만세 운동을 주도하다 일본 헌병대에 체포되었습니다.",
        "url": "https://archive.much.go.kr/archive/publicationRead/InfoDetailInqire.do?publicationId=PLCT_0000000208",
        "persona": "다정하고 따뜻하지만, 정의로운 의지를 가진 말투 '~해요'를 사용하세요."
    },
    "안중근": {
        "img": "anjunggeun.png",
        "fact": "1909년 10월 26일 하얼빈 역에서 이토 히로부미를 처단하여 대한 독립의 의지를 세계에 알렸습니다.",
        "url": "https://archive.much.go.kr/archive/publicationRead/InfoDetailInqire.do?publicationId=PLCT_0000000274",
        "persona": "논리적이고 침착하며 무게감 있는 말투 '~하오'를 사용하세요."
    },
    "윤봉길": {
        "img": "yunbonggil.png",
        "fact": "1932년 4월 29일 홍커우 공원에서 도시락 폭탄과 물통 폭탄을 던져 독립 의지를 증명했습니다.",
        "url": "https://archive.much.go.kr/archive/publicationRead/InfoDetailInqire.do?publicationId=PLCT_0000000274",
        "persona": "뜨거운 열정과 실천적인 리더십 말투 '~합시다!'를 사용하세요."
    }
}

# [.env 로컬 파일 자동 로드 헬퍼]
def load_env_file():
    for env_path in [".env", "../.env"]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

# [API 키 다중 취득 및 이중화 헬퍼]
def get_gemini_api_keys():
    """
    1차/2차 API 키를 수집하여 우선순위 리스트(Failover 순서)로 반환
    지원 환경 변수: VITE_GEMINI_API_KEY_1/2, GEMINI_API_KEY_1/2, GOOGLE_API_KEY_1/2 등
    """
    load_env_file()
    keys = []

    def _add_key(k_val):
        if k_val:
            cleaned = str(k_val).strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)

    # 1. 사이드바 수동 입력 키 (Key 1, Key 2)
    _add_key(st.session_state.get("user_gemini_key_1"))
    _add_key(st.session_state.get("user_gemini_key_2"))
    _add_key(st.session_state.get("user_gemini_key"))

    # 2. 이중화 환경 변수 후보군 (VITE_ 및 일반 접두사 지원)
    pair_names = [
        ("VITE_GEMINI_API_KEY_1", "VITE_GEMINI_API_KEY_2"),
        ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2"),
        ("GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2"),
    ]
    for k1, k2 in pair_names:
        _add_key(os.getenv(k1))
        _add_key(os.getenv(k2))

    # 3. Streamlit Secrets 확인
    try:
        if hasattr(st, "secrets"):
            for k1, k2 in pair_names:
                if k1 in st.secrets:
                    _add_key(st.secrets[k1])
                if k2 in st.secrets:
                    _add_key(st.secrets[k2])
            for s_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key", "GEMINI_API_KEY_SECONDARY"]:
                if s_name in st.secrets:
                    _add_key(st.secrets[s_name])
    except Exception:
        pass

    # 4. 단일 기본 환경 변수
    for s_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key", "GEMINI_API_KEY_SECONDARY"]:
        _add_key(os.getenv(s_name))

    return keys

# [2. UI 설정]
st.set_page_config(page_title="타임톡(Time-Talk)", page_icon="📜")
st.title("📜 타임톡(Time-Talk)")
st.caption("대한민국역사박물관 오픈아카이브 데이터 기반 AI 페르소나 챗봇")

# 사이드바 설정
st.sidebar.header("⚙️ 대화 설정")
char_name = st.sidebar.selectbox("대화할 인물을 선택하세요:", list(knowledge_base.keys()))
char_data = knowledge_base[char_name]

# API 키 연결 상태 및 페일오버(Failover) 지원 UI
current_keys = get_gemini_api_keys()
if len(current_keys) >= 2:
    st.sidebar.success(f"🟢 Gemini 연결됨 (활성 키 {len(current_keys)}개 / 자동 페일오버 가동)")
elif len(current_keys) == 1:
    st.sidebar.info("🟡 Gemini 연결됨 (단일 키 사용 중)")
    with st.sidebar.expander("➕ 보조 API 키 등록 (할당량 초과 방지)"):
        key2_in = st.text_input("보조 API 키 (Key #2):", type="password", placeholder="AIzaSy...", key="sub_key_in")
        if key2_in and st.button("보조 키 등록", key="btn_sub_save"):
            st.session_state["user_gemini_key_2"] = key2_in
            st.rerun()
else:
    st.sidebar.error("🔴 Gemini API 키 미등록")
    with st.sidebar.expander("🔑 API 키 직접 등록", expanded=True):
        key1_in = st.text_input("기본 API 키 (Key #1):", type="password", placeholder="AIzaSy...", key="key1_in")
        key2_in = st.text_input("보조 API 키 (Key #2 - 권장):", type="password", placeholder="AIzaSy...", key="key2_in")
        if st.button("API 키 저장", key="btn_keys_save"):
            if key1_in:
                st.session_state["user_gemini_key_1"] = key1_in
            if key2_in:
                st.session_state["user_gemini_key_2"] = key2_in
            st.rerun()

# 페르소나 변경 시 이전 대화 내용 자동 초기화
if "current_char" not in st.session_state:
    st.session_state.current_char = char_name

if st.session_state.current_char != char_name:
    st.session_state.messages = []
    st.session_state.current_char = char_name

# -------- Teacher worksheet generation ----------
st.sidebar.markdown("---")
if st.sidebar.button("📝 교사용 활동지 생성"):
    msgs = st.session_state.get("messages", [])
    def generate_worksheet(msgs, char_name, char_data):
        md = f"# {char_name} 학습 활동지\n\n"
        md += "## Ⅰ. 토론 주제\n"
        md += f"- 초등학생: {char_name}은 언제, 어디서 활약했나요?\n"
        md += f"- 중학생: {char_name}의 행동이 당시 한국에 어떤 영향을 미쳤나요?\n"
        md += f"- 고등학생: {char_name}의 사상을 현대에 어떻게 적용할 수 있나요?\n\n"
        md += "## Ⅱ. 빈칸 채우기 퀴즈\n"
        md += f"1. {char_name}은 ___년 ___월 ___일에 ___에서 활동을 시작했다.\n"
        md += f"2. {char_name}이 남긴 주요 업적은 ___이다.\n\n"
        md += f"### 참고 자료\n- 출처: [{char_data['url']}]({char_data['url']})\n"
        return md
    worksheet_md = generate_worksheet(msgs, char_name, char_data)
    st.subheader("📝 교사용 수업 활동지")
    st.markdown(worksheet_md)
    st.download_button(
        label="활동지 다운로드 (.md)",
        data=worksheet_md,
        file_name=f"{char_name}_worksheet.md",
        mime="text/markdown",
    )

st.sidebar.markdown("---")
if st.sidebar.button("🧹 대화 내용 초기화", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 사이드바 하단 여백 및 공공누리(공공데이터 오픈) 이미지
st.sidebar.markdown("<div style='height: 350px;'></div>", unsafe_allow_html=True)
try:
    if os.path.exists("img_opentype09.png"):
        st.sidebar.image("img_opentype09.png", width=160)
except Exception:
    pass

# 메인 UI 출력
col1, col2 = st.columns([1, 2])
with col1:
    try:
        st.image(char_data["img"], use_container_width=True)
    except Exception:
        st.warning("이미지 파일을 확인하세요.")
with col2:
    st.write(f"### {char_name} 의사/열사")
    st.info(f"**학습 페르소나:** {char_data['persona']}")

# [3. 위키미디어(Wikimedia) REST API 연동 및 엄격 검증 함수]
WIKI_HEADERS = {
    "User-Agent": "TimeTalkBot/1.0 (Educational Project; mailto:history@timetalk.org)"
}

EXCLUDE_PATTERNS = [
    "map", "location", "locator", "plan", "지도", "도면", "위치도",
    "logo", "icon", "symbol", "flag", "emblem", "mark",
    "stub", "disambig", "badge", "question_mark", ".svg"
]

STOP_KEYWORDS = {
    "없음", "none", "null", "인사", "안녕", "반가워", "질문", "대화", "감사", ""
}

def is_valid_historical_image(url, title, description):
    """지도, 로고, 다이어그램 등 부적합한 이미지 필터링"""
    combined = f"{url} {title} {description}".lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in combined:
            return False
    return True

def search_wikimedia_image_strict(keyword):
    """
    LLM이 추출한 단일 핵심 키워드(명사 1~2개)로만 위키미디어를 검색하고,
    키워드와 제목이 일치하는 신뢰도 높은 사료 이미지만 선별.
    일치하는 결과가 없거나 연관성이 낮으면 안전장치(None)로 Fallback.
    """
    if not keyword:
        return None

    cleaned_kw = keyword.strip()
    if cleaned_kw.lower() in STOP_KEYWORDS or len(cleaned_kw) < 2:
        return None

    encoded = urllib.parse.quote(cleaned_kw)

    # 1. 한국어 위키백과 REST 검색 API (최우선)
    wiki_url = f"https://ko.wikipedia.org/w/rest.php/v1/search/page?q={encoded}&limit=5"
    try:
        req = urllib.request.Request(wiki_url, headers=WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=4) as res:
            data = json.loads(res.read().decode("utf-8"))
            for page in data.get("pages", []):
                thumb = page.get("thumbnail")
                if not thumb or not thumb.get("url"):
                    continue
                raw_url = thumb["url"]
                page_title = page.get("title", "")
                desc = page.get("description", "") or ""

                # 지도, SVG, 로고 필터링
                if not is_valid_historical_image(raw_url, page_title, desc):
                    continue

                # 키워드와 페이지 제목의 일치성 검증 (제목이나 키워드에 상호 포함)
                norm_kw = re.sub(r"[\s\-_·]", "", cleaned_kw)
                norm_title = re.sub(r"[\s\-_·]", "", page_title)
                if norm_kw not in norm_title and norm_title not in norm_kw:
                    continue

                if raw_url.startswith("//"):
                    raw_url = "https:" + raw_url
                hd_url = re.sub(r"/\d+px-", "/500px-", raw_url)
                return {
                    "keyword": cleaned_kw,
                    "title": page_title,
                    "description": desc or f"{cleaned_kw} 관련 역사 자료",
                    "image_url": hd_url,
                    "page_url": f"https://ko.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
                }
    except Exception:
        pass

    # 2. 위키미디어 커먼즈(Commons) REST API (위키백과 부재 시 폴백)
    try:
        commons_url = f"https://commons.wikimedia.org/w/rest.php/v1/search/page?q={encoded}&limit=5"
        req = urllib.request.Request(commons_url, headers=WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=4) as res:
            data = json.loads(res.read().decode("utf-8"))
            for page in data.get("pages", []):
                thumb = page.get("thumbnail")
                if not thumb or not thumb.get("url"):
                    continue
                raw_url = thumb["url"]
                page_title = page.get("title", "")
                desc = page.get("description", "") or ""

                if not is_valid_historical_image(raw_url, page_title, desc):
                    continue

                clean_title = page_title.replace("File:", "").replace("_", " ")
                norm_kw = re.sub(r"[\s\-_·]", "", cleaned_kw)
                norm_title = re.sub(r"[\s\-_·]", "", clean_title)
                if norm_kw not in norm_title:
                    continue

                if raw_url.startswith("//"):
                    raw_url = "https:" + raw_url
                hd_url = re.sub(r"/\d+px-", "/500px-", raw_url)
                return {
                    "keyword": cleaned_kw,
                    "title": clean_title,
                    "description": "Wikimedia Commons 사료",
                    "image_url": hd_url,
                    "page_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page_title)}"
                }
    except Exception:
        pass

    return None

# [4. 제미나이(Gemini) 답변 생성 및 시각자료 키워드 추출 함수]
def get_persona_answer_with_keyword(char_name, user_question, char_data):
    """
    LLM 답변을 생성하며, 대화 맥락에서 우측 패널에 띄울 '가장 핵심적인 시각 자료 검색어(명사 1개)'를 분리 추출.
    반환: (clean_response_text, extracted_keyword)
    """
    api_keys = get_gemini_api_keys()
    if not api_keys:
        err_msg = f"[{char_name}의 답변] : Gemini API 키가 설정되지 않았습니다. 왼쪽 사이드바에 API 키를 입력하거나 .env 또는 Streamlit Secrets에 등록해주세요."
        return err_msg, None

    system_instruction = (
        f"당신은 대한민국의 역사 인물 '{char_name}'입니다. {char_data['persona']}\n"
        f"다음은 당신의 삶에 대한 역사적 사실입니다: {char_data['fact']}\n"
        f"답변 시 역사적 사실에 기반하여 성실히 답하세요.\n\n"
        f"【필수 시각자료 지침】\n"
        f"답변의 가장 마지막 줄에, 답변 내용 중 학생들에게 화면으로 보여줄 '가장 핵심적인 역사적 장소/유물/사건 명사 1개'를 반드시 다음 형식으로 1줄만 작성하세요:\n"
        f"[시각자료_검색어: 키워드]\n"
        f"- 예시 1: 답변에서 탑골공원을 언급한 경우 -> [시각자료_검색어: 탑골공원]\n"
        f"- 예시 2: 답변에서 하얼빈역을 언급한 경우 -> [시각자료_검색어: 하얼빈역]\n"
        f"- 예시 3: 답변에서 서대문형무소를 언급한 경우 -> [시각자료_검색어: 서대문형무소]\n"
        f"- 일상적 인사나 특정 시각 자료를 특정하기 어려울 경우 -> [시각자료_검색어: 없음]"
    )

    candidate_models = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.5-flash",
        "gemini-flash-latest"
    ]

    total_keys = len(api_keys)
    last_error = ""

    # 🔄 [Key Failover 루프]
    for key_idx, api_key in enumerate(api_keys, start=1):
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        print(f"[Gemini API] 시도 중: Key #{key_idx}/{total_keys} ({masked_key})")

        try:
            genai.configure(api_key=api_key)
        except Exception as config_err:
            print(f"[Gemini API Config Error] Key #{key_idx} 설정 오류: {config_err}")
            last_error = str(config_err)
            continue

        key_hit_quota = False

        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(user_question)

                raw_text = ""
                if hasattr(response, "text") and response.text:
                    raw_text = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    raw_text = response.candidates[0].content.parts[0].text

                if raw_text and raw_text.strip():
                    if key_idx > 1:
                        print(f"✅ [Failover Success] Key #{key_idx}로 자동 전환 후 [{model_name}] 모델 답변 생성 성공!")

                    # 키워드 태그 정규식 파싱
                    target_kw = None
                    kw_match = re.search(r"\[시각자료_검색어:\s*([^\]]+)\]", raw_text)
                    if kw_match:
                        extracted = kw_match.group(1).strip()
                        if extracted not in STOP_KEYWORDS and len(extracted) >= 2:
                            target_kw = extracted
                        clean_text = re.sub(r"\[시각자료_검색어:\s*[^\]]+\]", "", raw_text).strip()
                    else:
                        clean_text = raw_text.strip()

                    return clean_text, target_kw
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                if any(q in err_str.lower() for q in ["429", "quota", "resourceexhausted", "rate limit", "ratelimit"]):
                    key_hit_quota = True
                    print(f"⚠️ [Quota Limit Detected] Key #{key_idx} 할당량 초과 ({model_name}): {err_str[:60]}")
                    break
                continue

        if key_hit_quota and key_idx < total_keys:
            print(f"🔄 [Failover Triggered] Key #{key_idx} 할당량 초과 -> 보조 Key #{key_idx+1}로 자동 전환 재시도")
            continue

    if any(q in last_error.lower() for q in ["429", "quota", "resourceexhausted", "rate limit"]):
        err_msg = f"[{char_name}의 답변] : 구글 무료 계정의 요청 할당량(Quota)이 일시적으로 초과되었습니다. (등록된 {total_keys}개의 API 키 모두 소진) 약 20~30초 후에 다시 질문을 입력해 주세요."
    else:
        err_msg = f"[{char_name}의 답변] : API 오류 발생: {last_error}"
    return err_msg, None

# [5. 대화 세션 및 UI 렌더링 동기화]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력 (사료 이미지 또는 Fallback 안내)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_info"):
            img = message["image_info"]
            st.image(
                img["image_url"],
                caption=f"📷 [역사 사료: {img.get('keyword', '')}] {img['title']} ({img['description']})",
                use_container_width=True
            )
            st.caption(f"출처: [Wikimedia / 위키백과 사료 보기]({img['page_url']})")
        elif message.get("show_fallback"):
            st.caption("ℹ️ 관련 시각 자료 없음 (일치하는 공인 사료 사진 없음)")

# 질문 입력 및 응답 생성
if prompt := st.chat_input("역사에 대해 궁금한 점을 질문해보세요!"):
    # 1. 사용자 질문 렌더링 & 세션 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 메시지 (답변 생성 + 타겟 키워드 검색 + 실시간 동기화)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(f"{char_name} 님이 답변을 작성하고 시각 자료를 검증하고 있습니다..."):
            response_text, target_keyword = get_persona_answer_with_keyword(char_name, prompt, char_data)
            
            # 타겟 키워드로 엄격한 위키미디어 검색 수행
            wiki_image = search_wikimedia_image_strict(target_keyword) if target_keyword else None
            full_response = f"{response_text}\n\n🔗 [박물관 오픈아카이브 원문]({char_data['url']})"

        # 답변 텍스트 즉시 렌더링
        message_placeholder.markdown(full_response)

        # 이미지 또는 Fallback 안전장치 렌더링
        msg_record = {"role": "assistant", "content": full_response}
        if wiki_image:
            st.image(
                wiki_image["image_url"],
                caption=f"📷 [역사 사료: {wiki_image['keyword']}] {wiki_image['title']} ({wiki_image['description']})",
                use_container_width=True
            )
            st.caption(f"출처: [Wikimedia / 위키백과 사료 보기]({wiki_image['page_url']})")
            msg_record["image_info"] = wiki_image
        else:
            st.caption("ℹ️ 관련 시각 자료 없음 (일치하는 공인 사료 사진 없음)")
            msg_record["show_fallback"] = True

    # 세션 상태에 저장하여 새로고침 시에도 동일하게 동기화 유지
    st.session_state.messages.append(msg_record)
