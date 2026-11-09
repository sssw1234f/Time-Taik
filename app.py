import streamlit as st
import os
import io
import time
import urllib.request
import urllib.parse
import json
import re
import google.generativeai as genai
from qa_cache import get_cached_qa, save_cached_qa, get_cache_stats, sanitize_and_reset_cache, extract_keywords

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

# 학습자용 원클릭 질문 프리셋 (인물별 3가지 핵심 질문)
PRESET_QUESTIONS = {
    "유관순": [
        "📢 아우내 장터 만세 운동 당시의 상황을 들려주세요.",
        "🇰🇷 직접 태극기를 만들어 배포하셨을 때 어떤 심정이셨나요?",
        "⛓️ 서대문형무소 옥중에서도 만세를 멈추지 않으신 이유는 무엇인가요?"
    ],
    "안중근": [
        "🎯 하얼빈 역에서 이토 히로부미를 처단하신 이유가 무엇입니까?",
        "🤝 단지동맹(손가락 절단) 결성 당시의 결의를 말씀해 주세요.",
        "🕊️ 옥중에서 집필하신 《동양평화론》의 핵심 정신은 무엇인가요?"
    ],
    "윤봉길": [
        "💣 훙커우 공원 거사를 결심하게 된 계기는 무엇이었나요?",
        "🍱 물통 폭탄과 도시락 폭탄은 각각 어떤 용도였습니까?",
        "✉️ 떠나기 전 두 어린 아들에게 남기신 유언의 뜻을 알려주세요."
    ]
}

# [.env 로컬 파일 자동 로드 헬퍼]
def load_env_file():
    for env_path in [".env", "../.env", "Time-Taik-main/.env"]:
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

# [2. UI 설정 & 반응형 스타일링]
st.set_page_config(
    page_title="타임톡(Time-Talk) - 역사 AI 페르소나",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 모바일 뷰포트 최적화, 줄바꿈(keep-all) 및 불필요한 스크롤바 제거 CSS
st.markdown("""
<style>
/* 한글 단어 줄바꿈 및 메시지 박스 가독성 최적화 */
.stChatMessage, .stMarkdown, p, div {
    word-break: keep-all;
    overflow-wrap: break-word;
}

/* 프리셋 칩 버튼 스타일링 */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    border-radius: 18px;
    padding: 0.35rem 0.65rem;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1px solid #d1d5db;
    background-color: #f9fafb;
    color: #1f2937;
    transition: all 0.2s ease-in-out;
    white-space: normal;
    height: auto;
    min-height: 2.4rem;
}

div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    border-color: #3b82f6;
    color: #1d4ed8;
    background-color: #eff6ff;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.15);
}

/* 프로필 배지 및 카드 가독성 */
.hero-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* 사이드바 여백 정리 */
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* 부드러운 스크롤 */
html {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

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
            st.toast("✅ 보조 API 키가 성공적으로 등록되었습니다!", icon="🔑")
            st.rerun()
else:
    st.sidebar.warning("🔴 Gemini API 키 미등록 (체험 모드)")
    with st.sidebar.expander("🔑 API 키 직접 등록", expanded=True):
        key1_in = st.text_input("기본 API 키 (Key #1):", type="password", placeholder="AIzaSy...", key="key1_in")
        key2_in = st.text_input("보조 API 키 (Key #2 - 권장):", type="password", placeholder="AIzaSy...", key="key2_in")
        if st.button("API 키 저장", key="btn_keys_save"):
            if key1_in:
                st.session_state["user_gemini_key_1"] = key1_in
            if key2_in:
                st.session_state["user_gemini_key_2"] = key2_in
            st.toast("✅ API 키가 저장되었습니다!", icon="🔑")
            st.rerun()

# Q&A 지능형 캐시 상태 모니터링
st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Q&A 지능형 캐시 레이어")
cache_stats = get_cache_stats()
col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    st.metric("저장된 Q&A", f"{cache_stats['total_items']}개")
with col_c2:
    st.metric("절약된 호출", f"{cache_stats['total_hits']}회")
st.sidebar.caption("동일/유사 질의 시 LLM API 중복 호출을 차단하고 로컬 DB에서 0초 만에 응답합니다.")
if st.sidebar.button("🧹 캐시 정제 (검증 프리셋만 유지)", key="btn_clean_cache", use_container_width=True):
    sanitize_and_reset_cache()
    st.toast("✅ 오염 데이터 정제 완료! 9대 표준 프리셋으로 안전하게 복원되었습니다.", icon="🧹")
    st.rerun()

# 페르소나 변경 시 이전 대화 내용 자동 초기화
if "current_char" not in st.session_state:
    st.session_state.current_char = char_name

if st.session_state.current_char != char_name:
    st.session_state.messages = []
    st.session_state.current_char = char_name

# -------- Teacher worksheet generation (Modal & PDF) ----------
WORKSHEET_PROFILES = {
    "유관순": {
        "background": (
            "1919년 3월 1일 서울 탑골공원에서 시작된 3·1 만세 운동은 전 민족적 자주독립 투쟁으로 요원의 불길처럼 확산되었습니다. "
            "이화학당 고등과에 재학 중이던 유관순 열사는 일제의 휴교령 속에서도 위험을 무릅쓰고 고향 천안으로 내려가 직접 태극기를 만들어 주민들에게 나누어주고, "
            "1919년 4월 1일 아우내 장터 만세 시위를 주도하여 수천 명의 민중과 함께 대한독립 만세를 외쳤습니다. "
            "부모님이 일제 헌병의 총칼에 현장에서 순국하고 본인 또한 체포되어 서대문형무소에 수감되었으나, 모진 고문과 탄압 속에서도 옥중 만세 투쟁을 끝까지 멈추지 않으며 "
            "조국의 자주독립과 인간 존엄의 가치를 온몸으로 증명하였습니다."
        ),
        "elementary": "유관순 열사는 왜 안전한 서울에 머물지 않고 위험을 무릅쓰며 고향 천안 아우내 장터로 내려가 태극기를 나누어 주었을까요? 열사의 용기 있는 행동을 바탕으로 이야기해 봅시다.",
        "middle": "일제의 가혹한 총칼 앞에서도 굴하지 않고 옥중에서까지 만세를 외쳤던 유관순 열사의 결단은 당시 국내외 독립운동 전반과 민족의식에 어떤 파급효과를 가져왔을까요?",
        "high": "유관순 열사가 온몸으로 지켜낸 '자유와 독립'의 숭고한 가치를 오늘날 우리가 마주한 인권 신장, 민주주의 수호, 혹은 사회적 약자 연대의 관점에서 비판적으로 재해석해 보고, 현대 시민으로서 실천할 수 있는 용기는 무엇인지 논술해 봅시다.",
        "quiz_blanks": [
            "유관순 열사가 고향 천안에서 수천 명의 군중과 함께 독립 만세를 외쳤던 장소는 1919년 4월 1일의 ( ____________ ) 장터입니다.",
            "유관순 열사가 모진 고초를 겪으면서도 옥중 만세 투쟁을 이어가다 1920년 9월 28일 순국한 장소는 서울 ( ____________ ) 형무소입니다.",
            "유관순 열사가 재학하며 민족의식과 근대 지식을 길렀던 여성 교육 기관은 ( ____________ ) 학당입니다."
        ],
        "essay": "만약 내가 1919년 봄, 일제의 감시와 헌병대의 총칼이 삼엄했던 시대에 살고 있었다면, 아우내 장터의 만세 행렬 속에서 어떤 선택과 행동을 했을지 서술해 보세요."
    },
    "안중근": {
        "background": (
            "대한제국 말기, 일제는 을사늑약과 군대 해산을 강행하며 국권을 불법적으로 강탈하고 한반도를 식민지화하려 했습니다. "
            "안중근 의사는 연해주에서 동의단지회를 결성하여 손가락을 끊고 조국 독립을 맹세한 뒤, 1909년 10월 26일 중국 하얼빈 역에서 침략의 원흉인 이토 히로부미를 단호히 처단하였습니다. "
            "의사는 체포된 이후에도 대한국 의군 참모중장의 자격으로 일제의 불법 침략 죄상 15개 조목을 당당히 밝혔으며, "
            "순국 직전까지 뤼순 감옥에서 한·중·일 3국의 자주권 존중과 상호 협력을 통한 《동양평화론》을 저술하여 시대를 앞선 평화 사상을 인류에 남겼습니다."
        ),
        "elementary": "안중근 의사는 하얼빈 역에서 거사를 성공시킨 직후 도망치지 않고 러시아어로 '코레아 우라(대한독립 만세)'를 당당하게 외쳤습니다. 의사가 그 순간 당당할 수 있었던 이유는 무엇일까요?",
        "middle": "안중근 의사가 자신을 단순한 테러리스트나 암살자가 아닌 '대한국 의군 참모중장'이자 '전쟁 포로'로 대우할 것을 당당히 요구한 것은 국제법 및 독립운동사에서 어떤 역사적 의의를 지닐까요?",
        "high": "안중근 의사가 뤼순 감옥에서 주창한 《동양평화론》의 핵심 사상을 분석해 보고, 오늘날 군비 경쟁과 영토 분쟁이 지속되는 동북아시아의 평화 공존을 위해 이 사상을 어떻게 계승·적용할 수 있을지 비판적으로 논술해 봅시다.",
        "quiz_blanks": [
            "안중근 의사가 1909년 10월 26일 침략의 원흉 이토 히로부미를 처단한 역사적 장소는 중국 동북부의 ( ____________ ) 역입니다.",
            "안중근 의사가 11명의 동지들과 함께 왼손 넷째 손가락을 자르고 피로써 조국 독립을 맹세한 결사 조직은 ( ____________ ) 입니다.",
            "안중근 의사가 뤼순 옥중에서 집필하며 한·중·일 동양 3국의 공동 평화와 대등한 번영을 구상한 미완의 저서는 《 ( ____________ ) 》입니다."
        ],
        "essay": "만약 내가 1910년 뤼순 일본 관동도독부 법정에 선 안중근 의사의 변호인이었다면, 재판관들에게 안중근 의사의 거사에 대해 어떠한 법리적·역사적 변론을 펼쳤을지 서술해 보세요."
    },
    "윤봉길": {
        "background": (
            "1930년대 초, 일제가 만주사변과 상하이 사변을 잇따라 일으키며 대륙 침략을 노골화하던 시기, 대한민국 임시정부는 일제의 탄압과 재정난으로 침체기에 빠져 있었습니다. "
            "고향 예산에서 농촌 계몽 운동에 힘쓰다 망명한 청년 윤봉길 의사는 김구 선생의 한인애국단에 입단하여, 1932년 4월 29일 상하이 훙커우 공원의 일왕 생일 및 전승 기념식장에 폭탄을 투척하였습니다. "
            "이 거사는 일본 군부 수뇌부를 처단하고 침체되었던 임시정부의 활로를 열었으며, "
            "중국 국민당 장제스 총통이 '중국의 100만 대군도 하지 못한 일을 조선의 한 청년이 해냈다'고 격찬하며 임시정부를 전폭 지원하는 결정적 계기를 마련했습니다."
        ),
        "elementary": "윤봉길 의사는 어린 두 아들에게 '피가 있고 뼈가 있다면 반드시 조선을 위해 용감한 투사가 되어라'라는 유서를 남겼습니다. 자신의 목숨보다 나라를 먼저 생각했던 의사의 마음에 대해 이야기해 봅시다.",
        "middle": "훙커우 공원 거사에서 던진 물통 폭탄과 자결용 도시락 폭탄의 의미를 되새겨보고, 이 거사가 침체되어 있던 대한민국 임시정부와 한·중 연합 전선 형성에 미친 구체적 파급효과에 대해 토론해 봅시다.",
        "high": "농촌 계몽 운동(부흥원, 월진회)을 통해 실력 양성에 힘쓰던 윤봉길 의사가 마침내 무장 의열 투쟁이라는 길을 선택하게 된 시대적 배경을 비판적으로 검토하고, 목적을 위한 실천 방식의 변화가 갖는 역사적 의미를 논술해 봅시다.",
        "quiz_blanks": [
            "윤봉길 의사가 1932년 4월 29일 일제 수뇌부를 향해 폭탄을 투척한 상하이의 공원 명칭은 ( ____________ ) 공원(현 루쉰 공원)입니다.",
            "백범 김구 선생이 창설하여 윤봉길 의사와 이봉창 의사 등이 결사대로 활약한 대한민국 임시정부 산하 특무 조직은 ( ____________ ) 입니다.",
            "윤봉길 의사가 고향 충남 예산에서 청년들을 계몽하고 농촌 부흥을 이끌기 위해 창립했던 농민 운동 단체는 ( ____________ ) 회입니다."
        ],
        "essay": "만약 내가 25세 청년 윤봉길이었다면, 가족과의 평온한 일상과 조국의 독립이라는 거대한 갈림길 앞에서 어떤 고뇌와 선택을 내렸을지 솔직한 생각을 서술해 보세요."
    }
}

def get_worksheet_profile(char_name, char_data):
    return WORKSHEET_PROFILES.get(char_name, {
        "background": f"{char_name} 선생은 일제의 가혹한 침탈에 맞서 조국의 자주독립과 민족의 존엄을 지키기 위해 일생을 바치셨습니다. {char_data['fact']} 선생의 고결한 희생정신과 실천적 용기는 우리 역사의 든든한 초석이 되었습니다.",
        "elementary": f"{char_name} 선생이 조국을 위해 실천한 가장 대표적인 행동은 무엇이며, 그 행동에서 무엇을 느낄 수 있나요?",
        "middle": f"{char_name} 선생이 활동하던 시대의 어려움 속에서 내린 선택이 당시 민족사회에 미친 파급효과에 대해 토론해 봅시다.",
        "high": f"{char_name} 선생의 생애와 사상을 현대 사회가 직면한 갈등과 과제에 비추어 비판적으로 고찰해 보고, 오늘날 우리가 계승해야 할 핵심 가치는 무엇인지 논술해 봅시다.",
        "quiz_blanks": [
            f"{char_name} 선생의 주요 활동과 관련된 역사적 연도와 장소를 사료를 통해 찾아보세요: ( ____________ )",
            f"{char_name} 선생이 가담하거나 이끌었던 독립운동 조직의 명칭은 무엇인가요? ( ____________ )",
            f"{char_name} 선생이 평생을 바쳐 지키고자 했던 핵심 역사적 가치는 무엇인가요? ( ____________ )"
        ],
        "essay": f"만약 내가 {char_name} 선생이 살았던 격동의 시대에 태어났다면, 나라와 민족의 위기 앞에서 어떤 선택을 했을지 서술해 보세요."
    })

def generate_worksheet_markdown(char_name, char_data):
    p = get_worksheet_profile(char_name, char_data)
    doc = [
        f"## [{char_name}과 함께하는 역사 깊이 읽기]\n",
        f"**| 일시:** 202&nbsp;&nbsp;년 &nbsp;&nbsp;&nbsp;&nbsp;월 &nbsp;&nbsp;&nbsp;&nbsp;일 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **| 학년/반:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **| 이름:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **| 학습 인물:** {char_name} 의사/열사\n",
        "---\n",
        "### 1. 짚어보기 (핵심 역사 배경 요약)",
        f"> {p['background']}\n",
        "### 2. 수준별 토론 및 생각 나누기 (Discussion)",
        f"- **[초등학생 수준]** {p['elementary']}",
        "  - ✍️ *나의 생각:* _____________________________________________________________________________________\n",
        f"- **[중학생 수준]** {p['middle']}",
        "  - ✍️ *나의 생각:* _____________________________________________________________________________________\n",
        f"- **[고등학생 수준]** {p['high']}",
        "  - ✍️ *나의 생각:* _____________________________________________________________________________________\n",
        "### 3. 정밀 분석: 사료 및 팩트 퀴즈 (Quiz)",
        f"1. {p['quiz_blanks'][0]}",
        f"2. {p['quiz_blanks'][1]}",
        f"3. {p['quiz_blanks'][2]}",
        f"4. **[서술형 퀴즈]** {p['essay']}",
        "   - ✍️ *서술 답안:* _____________________________________________________________________________________\n",
        "### 4. 나만의 역사 한 줄 평 & 메타인지 (Reflection)",
        f"- **오늘 대화한 내용을 바탕으로 {char_name} 님에게 보내는 한 줄 평:**",
        "  > [                                                                                               ]",
        "- **내가 느낀 점을 서술하는 메타인지 기록장:**",
        "  - *오늘 대화를 통해 새롭게 깨달은 역사적 사실:* __________________________________________________",
        "  - *선생의 삶이 현대의 나에게 주는 교훈과 다짐:* __________________________________________________\n",
        "---\n",
        f"🔗 **[사료 근거 및 원문 확인하기]({char_data['url']})** (대한민국역사박물관 오픈아카이브)"
    ]
    return "\n".join(doc)

@st.cache_resource(show_spinner=False)
def get_korean_font_path():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "NanumGothic.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    local_f = "NanumGothic.ttf"
    if not os.path.exists(local_f):
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                with open(local_f, "wb") as f:
                    f.write(resp.read())
            if os.path.exists(local_f):
                return local_f
        except Exception:
            pass
    return None

def generate_worksheet_pdf(char_name, char_data):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        font_path = get_korean_font_path()
        if not font_path:
            return None

        font_name = "KoreanFont"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, font_path))

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=30,
            bottomMargin=30
        )

        title_style = ParagraphStyle(
            "TitleStyle",
            fontName=font_name,
            fontSize=15,
            leading=19,
            alignment=1,
            textColor=colors.HexColor("#111827")
        )
        info_style = ParagraphStyle(
            "InfoStyle",
            fontName=font_name,
            fontSize=8,
            leading=11,
            alignment=1,
            textColor=colors.HexColor("#4b5563")
        )
        h2_style = ParagraphStyle(
            "H2Style",
            fontName=font_name,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=5,
            spaceAfter=2
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            fontName=font_name,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#1f2937")
        )
        quote_style = ParagraphStyle(
            "QuoteStyle",
            fontName=font_name,
            fontSize=7.8,
            leading=11,
            textColor=colors.HexColor("#374151")
        )

        p = get_worksheet_profile(char_name, char_data)

        story = []
        story.append(Paragraph(f"[{char_name}과 함께하는 역사 깊이 읽기]", title_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"일시: 202&nbsp;&nbsp;년 &nbsp;&nbsp;월 &nbsp;&nbsp;일 &nbsp;&nbsp;|&nbsp;&nbsp; 학년/반: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp; 이름: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp; 대상 인물: {char_name} 의사/열사", info_style))
        story.append(Spacer(1, 3))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9ca3af"), spaceAfter=5))

        story.append(Paragraph("1. 짚어보기 (핵심 역사 배경 요약)", h2_style))
        story.append(Paragraph(p["background"], quote_style))
        story.append(Spacer(1, 3))

        story.append(Paragraph("2. 수준별 토론 및 생각 나누기 (Discussion)", h2_style))
        story.append(Paragraph(f"• <b>[초등학생]</b> {p['elementary']}", body_style))
        story.append(Paragraph("✍️ 나의 생각: ____________________________________________________________________________________", body_style))
        story.append(Spacer(1, 1))
        story.append(Paragraph(f"• <b>[중학생]</b> {p['middle']}", body_style))
        story.append(Paragraph("✍️ 나의 생각: ____________________________________________________________________________________", body_style))
        story.append(Spacer(1, 1))
        story.append(Paragraph(f"• <b>[고등학생]</b> {p['high']}", body_style))
        story.append(Paragraph("✍️ 나의 생각: ____________________________________________________________________________________", body_style))
        story.append(Spacer(1, 3))

        story.append(Paragraph("3. 정밀 분석: 사료 및 팩트 퀴즈 (Quiz)", h2_style))
        story.append(Paragraph(f"1. {p['quiz_blanks'][0]}", body_style))
        story.append(Paragraph(f"2. {p['quiz_blanks'][1]}", body_style))
        story.append(Paragraph(f"3. {p['quiz_blanks'][2]}", body_style))
        story.append(Paragraph(f"4. <b>[서술형 퀴즈]</b> {p['essay']}", body_style))
        story.append(Paragraph("✍️ 서술 답안: ____________________________________________________________________________________", body_style))
        story.append(Spacer(1, 3))

        story.append(Paragraph("4. 나만의 역사 한 줄 평 & 메타인지 (Reflection)", h2_style))
        story.append(Paragraph(f"• <b>오늘 대화한 내용을 바탕으로 {char_name} 님에게 보내는 한 줄 평:</b>", body_style))
        story.append(Paragraph("[ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ]", body_style))
        story.append(Paragraph("• <b>새롭게 깨달은 역사적 사실:</b> __________________________________________________________________", body_style))
        story.append(Paragraph("• <b>선생의 삶이 현대의 나에게 주는 교훈과 다짐:</b> ________________________________________________", body_style))

        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        return None

# 모달 다이얼로그 함수 정의 (Streamlit dialog 기능 활용)
if hasattr(st, "dialog"):
    @st.dialog("📝 교사용 수업 활동지", width="large")
    def open_worksheet_dialog(c_name, c_data):
        ws_md = generate_worksheet_markdown(c_name, c_data)
        st.markdown(ws_md)
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="📥 마크다운 다운로드 (.md)",
                data=ws_md,
                file_name=f"{c_name}_worksheet.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_btn2:
            pdf_bytes = generate_worksheet_pdf(c_name, c_data)
            if pdf_bytes:
                st.download_button(
                    label="📄 PDF 다운로드 (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{c_name}_worksheet.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("PDF 생성 모듈 로딩 중...")
elif hasattr(st, "experimental_dialog"):
    @st.experimental_dialog("📝 교사용 수업 활동지", width="large")
    def open_worksheet_dialog(c_name, c_data):
        ws_md = generate_worksheet_markdown(c_name, c_data)
        st.markdown(ws_md)
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="📥 마크다운 다운로드 (.md)",
                data=ws_md,
                file_name=f"{c_name}_worksheet.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_btn2:
            pdf_bytes = generate_worksheet_pdf(c_name, c_data)
            if pdf_bytes:
                st.download_button(
                    label="📄 PDF 다운로드 (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{c_name}_worksheet.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("PDF 생성 모듈 로딩 중...")
else:
    def open_worksheet_dialog(c_name, c_data):
        st.session_state["show_worksheet_dialog"] = True

# 사이드바 활동지 생성 버튼
st.sidebar.markdown("---")
if st.sidebar.button("📝 교사용 활동지 생성", use_container_width=True):
    open_worksheet_dialog(char_name, char_data)

# 다이얼로그 미지원 구버전 환경을 위한 안전 폴백
if st.session_state.get("show_worksheet_dialog", False):
    with st.expander("📝 [팝업창] 교사용 수업 활동지", expanded=True):
        ws_md = generate_worksheet_markdown(char_name, char_data)
        st.markdown(ws_md)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 마크다운 (.md)", ws_md, f"{char_name}_worksheet.md", "text/markdown", use_container_width=True)
        with c2:
            pdf_bytes = generate_worksheet_pdf(char_name, char_data)
            if pdf_bytes:
                st.download_button("📄 PDF (.pdf)", pdf_bytes, f"{char_name}_worksheet.pdf", "application/pdf", use_container_width=True)
        if st.button("❌ 닫기"):
            st.session_state["show_worksheet_dialog"] = False
            st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🧹 대화 내용 초기화", use_container_width=True):
    st.session_state.messages = []
    st.toast("대화 기록이 초기화되었습니다.", icon="🧹")
    st.rerun()

# 사이드바 공공누리(공공데이터 오픈) 이미지 및 안내
st.sidebar.markdown("---")
try:
    if os.path.exists("img_opentype09.png"):
        st.sidebar.image("img_opentype09.png", width=140, caption="공공누리 제1유형: 출처표시")
except Exception:
    pass

# 메인 인물 소개 카드 UI 출력
col1, col2 = st.columns([1, 2.2], gap="medium")
with col1:
    try:
        st.image(char_data["img"], use_container_width=True)
    except Exception:
        st.warning(f"📷 {char_name} 이미지 로딩 중")
with col2:
    st.write(f"### {char_name} 의사/열사")
    st.info(f"**학습 페르소나:** {char_data['persona']}")
    st.caption(f"**기본 사료:** {char_data['fact']}")

# [3. 위키미디어(Wikimedia) REST API 연동 및 엄격 검증 함수 - 캐싱 적용]
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

@st.cache_data(show_spinner=False, ttl=3600)
def search_wikimedia_image_strict(keyword):
    """
    LLM이 추출한 단일 핵심 키워드로 위키미디어를 검색하고,
    키워드와 제목이 일치하는 신뢰도 높은 사료 이미지만 선별.
    캐싱(TTL 1시간)을 통해 반복 호출 방지 및 초고속 응답 보장.
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

                # 키워드와 페이지 제목의 일치성 검증 (상호 포함)
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

# [4. 사료 기반 지능형 지식 코퍼스(Corpus) 및 IR 검색 엔진]
HISTORICAL_KNOWLEDGE_CORPUS = {
    "유관순": [
        {
            "id": "school_life",
            "topic": "학창 시절 및 이화학당",
            "primary": ["이화학당", "이화", "학당", "학교", "공부", "배움", "과목", "신학문", "프라이", "월터", "정동", "기숙사", "학창", "선생님", "스승", "친구", "동창"],
            "secondary": ["어릴", "어린", "유년", "수업", "성경", "장학생", "서울"],
            "answer": (
                "저는 1916년 충남 천안을 떠나 서울 정동 이화학당 보통과 3학년에 교비 장학생으로 편입하였고, 이후 고등과에 진학해 신학문과 성경을 배웠어요. "
                "프라이 교장선생님과 월터 선생님의 따뜻한 가르침 속에서 성실히 배우며 빼앗긴 조국의 현실에 깊이 눈뜨게 되었답니다. "
                "이화학당 교정에서 친구들과 함께 품었던 자주독립의 기도가 훗날 아우내 만세 운동의 굳건한 디딤돌이 되었어요."
            ),
            "keyword": "유관순"
        },
        {
            "id": "flag_making",
            "topic": "태극기 제작 및 비밀 배포",
            "primary": ["태극기", "국기", "만들", "제작", "그렸", "배포", "인쇄", "나누", "기름종이", "목판", "도장"],
            "secondary": ["밤샘", "등불", "비밀", "준비", "가방", "치마"],
            "answer": (
                "일제의 임시 휴교령으로 서울에서 고향 천안으로 내려온 뒤, 동포들의 손에 쥐여줄 태극기를 한 땀 한 땀 직접 그렸어요. "
                "사촌 언니 유예도와 함께 어두운 등불 아래에서 기름종이에 밤을 새워 만든 태극기를 품에 안고 인근 마을을 돌며 배포했습니다. "
                "장터에 펄럭이던 그 날의 벅찬 감동은 지금도 가슴을 뜨겁게 울린답니다."
            ),
            "keyword": "태극기"
        },
        {
            "id": "aunae_movement",
            "topic": "아우내 장터 만세 운동 및 매봉산 봉화",
            "primary": ["아우내", "장터", "3·1", "삼일", "만세", "시위", "천안", "매봉산", "봉화", "4월 1일", "군중", "장날", "외침", "행진"],
            "secondary": ["함성", "연설", "군청", "헌병", "총칼", "탄압", "체포", "상황"],
            "answer": (
                "1919년 4월 1일(음력 3월 1일) 아우내 장터 만세 운동은 거사 전날 밤 매봉산에 봉화를 올려 인근 24개 동리에 신호를 보내며 시작되었어요. "
                "장날 정오 3천여 명의 군중이 함께 모여 조국의 자주독립을 목놓아 외쳤던 거룩한 날이었답니다. "
                "일제 헌병의 무차별 사격 앞에서도 물러서지 않고 태극기를 나누며 끝까지 만세를 외쳤습니다."
            ),
            "keyword": "탑골공원"
        },
        {
            "id": "family_sacrifice",
            "topic": "부모님 순국 및 가족사",
            "primary": ["가족", "부모", "부모님", "아버지", "어머니", "유중권", "이소제", "오빠", "유우석", "동생", "유관옥", "집안", "피"],
            "secondary": ["희생", "슬픔", "형제", "친척", "가문"],
            "answer": (
                "아우내 장터 만세 시위 현장에서 저의 아버지 유중권 님과 어머니 이소제 님께서 일제 헌병의 흉탄과 칼날에 현장에서 순국하셨어요. "
                "부모님을 한날한시에 잃은 슬픔은 가슴이 찢어지듯 아팠지만, 부모님의 거룩한 희생이 결코 헛되지 않도록 끝까지 조국의 독립을 향한 걸음을 멈추지 않았습니다."
            ),
            "keyword": "유관순"
        },
        {
            "id": "prison_resistance",
            "topic": "서대문형무소 수감 및 옥중 만세 투쟁",
            "primary": ["서대문형무소", "형무소", "감옥", "수감", "옥중", "여옥사", "8호", "감방", "고문", "수의", "붉은", "벽돌", "어윤희", "심명철", "임명애"],
            "secondary": ["취조", "매질", "통증", "어둠", "간수", "1주년"],
            "answer": (
                "공주지방법원을 거쳐 서대문형무소 여옥사 8호 감방에 수감되었을 때 입었던 붉은색 수의와 차가운 시멘트 바닥의 기억은 지금도 생생해요. "
                "일제는 모진 고문과 탄압으로 우리의 입을 막으려 했지만, 1920년 3월 1일 옥중에서도 동지들과 함께 감옥이 흔들리도록 만세를 불렀습니다. "
                "육신은 감옥에 갇혔을지언정 자주독립을 향한 우리의 영혼은 결코 꺾을 수 없었답니다."
            ),
            "keyword": "서대문형무소"
        },
        {
            "id": "martyrdom_spirit",
            "topic": "순국, 최후의 유언 및 민족정신",
            "primary": ["순국", "사망", "유언", "마지막", "1920년", "9월 28일", "나이", "18세", "손톱", "정신", "마음", "교훈", "청소년", "학생", "용기", "희망", "단결", "꺾이지"],
            "secondary": ["고통", "영혼", "후손", "현대", "자유", "평화", "존엄"],
            "answer": (
                "모진 고문의 후유증으로 1920년 9월 28일 서대문형무소 차가운 감방에서 18세의 나이로 순국하였습니다. "
                "독립운동에서 가장 큰 힘이 되었던 것은 바로 이름 모를 수많은 동포들의 '꺾이지 않는 마음'이었어요. "
                "'내 손톱이 빠져나가고 내 귀와 코가 잘려도 나라를 잃은 고통만은 견딜 수 없다'는 마음으로 조국을 품었습니다. 여러분도 정의와 연대의 가치를 마음에 새겨주세요."
            ),
            "keyword": "3·1 운동"
        }
    ],
    "안중근": [
        {
            "id": "weapon_and_harbin",
            "topic": "하얼빈 의거 및 사용 무기/권총",
            "primary": ["무기", "권총", "총", "총기", "브라우닝", "fn", "총번", "262336", "탄환", "총알", "총성", "발사", "십자", "덤덤탄", "의거", "거사", "하얼빈", "이토", "히로부미", "처단", "저격", "암살", "플랫홈", "코레아 우라", "만세", "10월 26일"],
            "secondary": ["장전", "격발", "저격총", "방아쇠", "단죄", "심판", "역사적"],
            "answer": (
                "1909년 10월 26일 하얼빈 역 거사 당시 본관이 사용한 무기는 벨기에제 'FN 브라우닝 M1900 8연발 반자동 권총(총번 262336)'이었소. "
                "침략의 원흉 이토 히로부미를 확실히 단죄하기 위해 탄두 끝에 십자(+) 흠을 낸 탄환을 장전하여 이토에게 3발을 명중시켰고, 이어 침략에 동조한 일제 수뇌부를 향해 4발을 명중시켰소. "
                "총성이 멎은 직후 러시아어로 '코레아 우라(대한독립 만세)'를 외치며 당당히 체포되었소."
            ),
            "keyword": "하얼빈역"
        },
        {
            "id": "comrades_and_plan",
            "topic": "하얼빈 거사 동지 및 협조자",
            "primary": ["동지", "협조자", "누구와", "함께", "우덕순", "유동하", "조도선", "최재형", "이석산", "채가구", "채가구역", "대동공보", "계획", "준비", "동역자"],
            "secondary": ["연해주", "블라디보스토크", "여비", "자금", "거사계획"],
            "answer": (
                "하얼빈 의거는 본관 혼자만의 힘으로 이룬 것이 아니오. 연해주에서 거사 자금과 무기를 지원해 준 대동공보사의 최재형 선생, "
                "그리고 생사를 함께하기로 맹세하고 채가구 역과 하얼빈 역에서 거사를 분담했던 우덕순, 조도선, 유동하 동지들의 피끓는 연대와 헌신이 있었기에 "
                "침략의 원흉을 향해 정의의 총성을 울릴 수 있었소."
            ),
            "keyword": "안중근"
        },
        {
            "id": "youth_and_marksmanship",
            "topic": "유년기, 아명(응칠), 사격 및 무예",
            "primary": ["어릴", "어린", "유년", "아명", "응칠", "무예", "사격", "사냥", "포수", "명사수", "말타기", "활쏘기", "성격", "고향", "해주", "청계동"],
            "secondary": ["성장", "소년", "호연지기", "서당", "담력"],
            "answer": (
                "본관의 아명은 가슴과 배에 북두칠성을 닮은 일곱 개의 점이 있어 응칠(應七)이라 불렸소. "
                "황해도 신천 청계동에서 글공부도 하였으나, 말타기와 활쏘기, 사냥을 즐기며 백두산 포수들과 어울려 뛰어난 명사수로 이름을 날렸소. "
                "이때 기른 담대한 배포와 정확한 사격 솜씨는 훗날 하얼빈 역 플랫홈에서 단 세 발로 침략의 원흉을 명중시키는 결정적 밑거름이 되었소."
            ),
            "keyword": "안중근"
        },
        {
            "id": "education_movement",
            "topic": "삼흥학교·돈의학교 교육 구국 및 국채보상운동",
            "primary": ["학교", "교육", "삼흥", "삼흥학교", "돈의", "돈의학교", "계몽", "국채보상", "인재", "가르치", "남포", "배움"],
            "secondary": ["계몽운동", "사재", "설립", "담배", "모금"],
            "answer": (
                "나라의 주권을 지키려면 무엇보다 백성을 깨우치고 민족의 인재를 길러내야 한다고 확신하였소. "
                "그리하여 사재를 털어 남포에 삼흥학교(三興學校)를 설립하고 돈의학교를 인수하여 민족 교육에 온 힘을 쏟았으며, 1907년에는 담배를 끊고 국채보상운동에 앞장섰소. "
                "참된 실력과 학문이야말로 나라를 지탱하는 굳건한 주춧돌이오."
            ),
            "keyword": "안중근"
        },
        {
            "id": "danji_dongmaeng",
            "topic": "단지동맹 (손가락 절단) 및 의병 결의",
            "primary": ["단지동맹", "단지", "손가락", "절단", "마디", "피", "맹세", "태극기", "연해주", "동의회", "의병", "대한독립", "결사", "1909"],
            "secondary": ["동포", "결의", "12명", "충의", "맹약"],
            "answer": (
                "1909년 봄 연해주에서 11명의 우국지사 동지들과 함께 왼손 넷째 손가락 첫마디를 자르고 피로써 태극기 위에 '대한독립' 네 글자를 새겼소. "
                "조국의 독립과 동양 평화를 위해 목숨을 바치겠다는 죽음을 넘어선 맹세였으며, "
                "사리사욕을 버리고 대동단결하는 민족의 기개가 그 붉은 피에 서려 있었소."
            ),
            "keyword": "단지동맹"
        },
        {
            "id": "mother_jomaria",
            "topic": "어머니 조마리아 여사의 편지와 명주 수의",
            "primary": ["어머니", "조마리아", "편지", "수의", "명주", "가족", "부모", "아내", "김아려", "아들", "분도", "준생", "효도", "항소"],
            "secondary": ["모성", "모친", "절개", "눈물", "당부"],
            "answer": (
                "뤼순 옥중에 갇혔을 때, 어머니 조마리아 여사께서는 '네가 항소를 하여 목숨을 구걸한다면 그것은 어미를 욕되게 하는 일이다. 너의 죽음은 너 한 사람의 것이 아니라 조선인 전체의 공분을 짊어진 것이니, 대의에 죽는 것이 어미에 대한 효도다'라는 준엄한 편지와 함께 손수 지으신 명주 수의를 보내오셨소. "
                "어머니의 그 위대한 모성과 가르침이 본관의 마음에 영원한 등불이 되었소."
            ),
            "keyword": "안중근"
        },
        {
            "id": "east_asia_peace",
            "topic": "《동양평화론》 집필 및 평화 사상",
            "primary": ["동양평화론", "동양평화", "평화론", "사상", "평화", "저서", "책", "글", "뤼순", "옥중", "집필", "한중일", "동양3국", "공동", "화합"],
            "secondary": ["군대", "은행", "공존", "침략", "군국주의"],
            "answer": (
                "뤼순 감옥 옥중에서 미완성으로 남긴 저서 《동양평화론》의 핵심은 한·중·일 세 나라가 서로의 자주권을 철저히 존중하며 평등하게 연대하자는 것이었소. "
                "뤼순을 중립항으로 삼아 동양 평화회의를 열고, 공동 은행과 연합군을 창설하여 서세동점의 외세에 맞서 영구적인 평화를 수호하자는 혜안을 담았소. "
                "진정한 독립이 전제될 때 비로소 참된 평화가 열리는 법이오."
            ),
            "keyword": "동양평화론"
        },
        {
            "id": "trial_and_martyrdom",
            "topic": "뤼순 재판, 이토의 15대 죄상, 순국 및 유언",
            "primary": ["재판", "법정", "죄상", "15개", "15조목", "이토죄상", "관동도독부", "사형", "순국", "유언", "하얼빈공원", "유해", "무덤", "3월 26일", "최후"],
            "secondary": ["명성황후", "을사늑약", "침략", "변호사", "사형집행"],
            "answer": (
                "일제의 불법 법정에서 본관은 일개 자객이 아니라 대한국 의군 참모중장으로서 교전 단체의 자격으로 명성황후 시해와 을사늑약 강제 체결 등 이토의 15가지 죄상을 준엄하게 꾸짖었소. "
                "1910년 3월 26일 순국하기 전 '나의 뼈를 하얼빈 공원에 묻어두었다가 국권이 회복되거든 고국으로 반환해 다오'라는 유언을 남겼소. "
                "대한독립의 소리가 천국에 들려오면 본관은 마땅히 춤을 추며 만세를 부를 것이오."
            ),
            "keyword": "안중근"
        }
    ],
    "윤봉길": [
        {
            "id": "bomb_and_hongkou",
            "topic": "훙커우 공원 거사 및 폭탄(물통, 도시락, 무기)",
            "primary": ["무기", "폭탄", "도시락", "물통", "도시락폭탄", "물통폭탄", "형태", "모양", "종류", "거사", "훙커우", "홍커우", "공원", "루쉰공원", "상하이", "상해", "4월 29일", "천장절", "시라카와", "노무라", "단상", "투척"],
            "secondary": ["위장", "자결", "도쿄", "작렬", "파편", "수뇌부", "처단"],
            "answer": (
                "1932년 4월 29일 상하이 훙커우 공원 거사 당시 사용된 무기는 겉보기에 일반적인 물통과 도시락 모양으로 정밀 위장된 특수 폭탄이었습니다! "
                "침략의 수괴인 일제 군관 수뇌부를 단죄하기 위해 단상으로 힘차게 투척한 것은 가죽 끈이 달린 '물통 폭탄'이었으며, "
                "거사 직후 일제의 굴욕적인 체포를 피하고 민족의 자존을 지키기 위해 준비했던 것은 자결용 '도시락 폭탄'이었습니다. "
                "투척된 물통 폭탄이 단상 중앙에서 맹렬히 작렬하여 시라카와 대장과 노무라 중장 등 일제 침략의 주범들을 단숨에 쓰러뜨렸습니다!"
            ),
            "keyword": "루쉰공원"
        },
        {
            "id": "kimgu_and_watch",
            "topic": "백범 김구와의 만남, 한인애국단, 회중시계 교환",
            "primary": ["김구", "백범", "한인애국단", "만남", "회중시계", "시계", "교환", "임시정부", "선서", "태극기앞", "망명", "장부출가생불환", "떠나", "이유", "계기"],
            "secondary": ["상하이", "선서문", "수류탄", "권총", "작별", "6원", "2원"],
            "answer": (
                "'장부가 집을 나가 뜻을 이루기 전에는 살아 돌아오지 않는다(장부출가생불환)'라는 글을 남기고 상하이로 건너가 백범 김구 선생님을 뵈었습니다. "
                "임시정부의 침체를 깨뜨리고 민족의 자주독립 의지를 세계만방에 알리고자 한인애국단에 입단하여 목숨을 바치기로 맹세했지요. "
                "거사 당일 아침 김구 선생님과 붉은색 회중시계를 맞바꾸며 '제 시계는 6원짜리이고 선생님 시계는 2원짜리인데, 저는 이제 1시간밖에 쓸 데가 없으니 바꾸시지요'라며 마지막 작별을 고했던 순간이 지금도 생생합니다!"
            ),
            "keyword": "매헌윤봉길의사기념관"
        },
        {
            "id": "rural_enlightenment",
            "topic": "농촌 계몽 운동, 야학당, 농민독본, 월진회",
            "primary": ["야학", "야학당", "농촌", "계몽", "농민독본", "월진회", "부흥", "농민", "가르치", "교육", "책", "오치서숙", "예산", "고향", "청년"],
            "secondary": ["자립", "한글", "민족혼", "부흥운동", "계몽가", "문맹"],
            "answer": (
                "19세 때 고향 충남 예산에서 가난한 농촌 청년들을 깨우치기 위해 야학당을 열어 한글과 민족의식을 가르쳤습니다. "
                "또한 농민들의 단결과 경제적 자립을 이끌고자 3권으로 구성된 《농민독본》을 직접 저술하였고, '달이 나아가듯 매일 전진하자'는 뜻의 '월진회(月進會)'를 창립해 농촌 부흥 운동에 청춘을 바쳤습니다. "
                "백성이 배움을 통해 깨어나야만 나라의 독립도 가능하다는 확고한 신념 때문이었습니다!"
            ),
            "keyword": "매헌윤봉길의사기념관"
        },
        {
            "id": "will_to_sons",
            "topic": "두 아들(모순, 담)에게 남긴 유언시",
            "primary": ["유언", "아들", "모순", "담", "편지", "유언시", "가족", "아내", "배용순", "부정", "당부", "피와 뼈", "투사"],
            "secondary": ["아버지", "자식", "슬픔", "희생", "마지막말"],
            "answer": (
                "상하이를 떠나며 사랑하는 두 어린 아들 모순(종)과 담에게 눈물의 유언시를 남겼습니다. "
                "'너희에게 피와 뼈가 있다면 반드시 조선을 위해 용감한 투사가 되어라. 태극 깃발을 높이 날리고 나의 빈 무덤을 찾아와 한 잔의 술을 부어 놓으라'는 글이었지요. "
                "아비로서 아이들의 곁을 지켜주지 못하는 슬픔을 억누르고, 조국의 자유와 후손들의 찬란한 미래를 위해 기꺼이 목숨을 던진 청년의 맹세였습니다!"
            ),
            "keyword": "매헌윤봉길의사기념관"
        },
        {
            "id": "kanazawa_martyrdom",
            "topic": "체포, 가나자와 순국, 십자가 형틀",
            "primary": ["순국", "가나자와", "사형", "십자가", "형틀", "최후", "마지막", "체포", "압송", "총살", "12월 19일", "미간", "총탄"],
            "secondary": ["오사카", "일본", "군법회의", "결연", "눈빛"],
            "answer": (
                "거사 직후 현장에서 체포되어 일본으로 압송된 뒤, 1932년 12월 19일 가나자와 육군 공병작업장에서 십자가 형틀에 묶여 총살형을 당했습니다. "
                "그러나 미간에 총탄을 맞는 마지막 순간까지 일제의 불의에 결코 고개 숙이지 않고 당당한 눈빛으로 조국의 영원한 독립을 응시했습니다! "
                "그날의 거사는 중국 장제스 총통이 '중국의 100만 대군도 하지 못한 일을 조선의 한 청년이 해냈다'며 대한민국 임시정부를 전폭 지원하는 결정적 계기가 되었습니다!"
            ),
            "keyword": "윤봉길"
        },
        {
            "id": "youth_spirit",
            "topic": "청년 정신, 행동하는 양심, 현대 청년들에게",
            "primary": ["청년", "정신", "실천", "용기", "열정", "메시지", "꺾이지", "희망", "양심", "교훈", "오늘날", "현대"],
            "secondary": ["동포", "투혼", "행동", "불꽃", "새벽"],
            "answer": (
                "청년의 피가 끓는 한 우리 민족은 결코 죽지 않습니다! "
                "탁상 위의 공리공론보다 불의에 맞서 행동하는 용기야말로 역사를 바꾸는 진정한 힘입니다. "
                "피와 뼈가 있다면 불의에 타협하지 않고 용감하게 전진할 때 반드시 자유의 새벽이 올 것입니다. "
                "오늘을 살아가는 청년 여러분도 시대의 부름에 응답하는 정의로운 주역이 되어 당당히 전진하십시오!"
            ),
            "keyword": "윤봉길"
        }
    ]
}


def retrieve_historical_persona_knowledge(char_name: str, question: str):
    """
    사료 기반 지능형 정보 검색(IR) 엔진:
    질문의 핵심 명사와 형태소 토큰을 정밀 분석하여, 인물별 사료 코퍼스(Corpus)에서
    가장 높은 가중치 점수를 획득한 정밀 지식 청크를 동적으로 추출합니다.
    반환: (answer_text, target_keyword, matched: bool)
    """
    corpus = HISTORICAL_KNOWLEDGE_CORPUS.get(char_name, [])
    tokens = extract_keywords(question)
    q_lower = question.lower()

    best_chunk = None
    best_score = 0.0

    for chunk in corpus:
        score = 0.0
        # 1. Primary keywords (가중치 3.0점)
        for p in chunk.get("primary", []):
            if p in q_lower:
                score += 3.0
            else:
                for t in tokens:
                    if p == t or (len(p) >= 2 and (p in t or t in p)):
                        score += 3.0
                        break

        # 2. Secondary keywords (가중치 1.5점)
        for s in chunk.get("secondary", []):
            if s in q_lower:
                score += 1.5
            else:
                for t in tokens:
                    if s == t or (len(s) >= 2 and (s in t or t in s)):
                        score += 1.5
                        break

        if score > best_score:
            best_score = score
            best_chunk = chunk

    # 매칭 임계치: 핵심 키워드 1개 이상(score >= 2.5) 매칭 시 해당 지식 청크 반환
    if best_chunk and best_score >= 2.5:
        return best_chunk["answer"], best_chunk["keyword"], True

    return None, None, False


def generate_contextual_persona_fallback(char_name, question, char_data):
    """
    외부 LLM API 호출 불가(키 오류, 쿼터 소진, 네트워크 지연) 시에도
    질문 맥락을 분석하여 역사 사료 기반의 자연스러운 페르소나 답변 및 시각자료를 생성하는 3중 안전장치.
    반환: (text, kw, is_llm=False, is_out_of_scope: bool)
    """
    q = question.lower()
    
    # 1. 범위 외(Out-of-Scope) 질문 키워드 감지 (날씨, 음식, 코딩, 연예 등)
    out_of_scope_keywords = [
        "날씨", "비 오", "눈 오", "기온", "점심", "저녁", "아침", "메뉴", "밥", "식사", "배고파", "맛집", "국밥",
        "코딩", "파이썬", "자바", "프로그래밍", "코드", "버그", "주식", "코인", "비트코인",
        "아이돌", "연예인", "노래", "영화", "드라마", "게임", "축구", "야구"
    ]
    is_out_of_scope = any(w in q for w in out_of_scope_keywords)

    if is_out_of_scope:
        if char_name == "유관순":
            text = (
                "현대의 일상이나 소소한 잡담도 소중하지만, 지금은 빼앗긴 나라를 되찾기 위해 온 겨레가 피 흘렸던 그 날의 외침을 먼저 돌아보았으면 해요. "
                "사소한 잡담에 마음을 두기보다 우리 조국을 지켜낸 숭고한 역사의 의미를 함께 나누어 보아요."
            )
        elif char_name == "안중근":
            text = (
                "하늘의 날씨나 일상의 소소한 일을 살피는 것도 중요하지만, 우리 민족의 마음에 드리운 일제의 그림자를 거두는 것이 더 급선무요. "
                "잡담에 마음을 쓰기보다 오늘의 역사를 돌아보고 조국과 평화의 참뜻을 깊이 헤아려 보시오."
            )
        else:  # 윤봉길
            text = (
                "점심 메뉴나 일상의 한가로운 이야기를 나눌 때가 아닙니다! 지금 우리에게 가장 절실한 것은 조국의 독립과 민족의 자주권입니다. "
                "청년들이여, 헛된 잡담에 귀중한 시간을 쏟기보다 불타는 애국심과 실천적 용기로 역사의 부름에 응답합시다!"
            )
        return text, "없음", False, True

    # 2. 사료 기반 지능형 정보 검색(IR) 엔진 가동
    matched_answer, matched_kw, is_matched = retrieve_historical_persona_knowledge(char_name, question)
    if is_matched:
        return matched_answer, matched_kw, False, False

    # 3. 임계치 미달 시 질문 맥락을 지능적으로 수용하는 페르소나 적응형(Adaptive) 응답
    if char_name == "유관순":
        text = (
            "질문해 주신 그 시대의 상황과 아픔에 대해 말씀드리자면, 일제의 무자비한 탄압 아래 온 겨레가 크나큰 고통을 겪고 있었어요. "
            "하지만 우리는 결코 절망하지 않고 1919년 3·1 만세 운동과 아우내 장터의 외침으로 당당히 맞섰답니다. "
            "조국을 지켜낸 숭고한 역사와 독립운동에 대해 어떤 점이 더 궁금하신가요?"
        )
        kw = "유관순"
    elif char_name == "안중근":
        text = (
            "질문하신 일제의 침략과 당시의 엄혹한 정세에 대해 말씀드리자면, 일제는 우리의 국권뿐 아니라 동양 전체의 평화를 무참히 유린하고 있었소. "
            "본관은 이에 맞서 의병 투쟁과 1909년 하얼빈 의거로 침략의 원흉을 단죄하고 민족의 자주독립 의지를 세계에 천명하였소. "
            "조국의 주권을 되찾기 위한 여정에 대해 무엇이든 더 물어보시오."
        )
        kw = "안중근"
    else:  # 윤봉길
        text = (
            "문의하신 일제의 압제와 당시의 암담했던 시대상에 대해 말씀드리자면, 빼앗긴 조국에서 우리 청년들과 백성들은 자유를 잃은 채 신음하고 있었습니다. "
            "그렇기에 가만히 주저앉아 있을 수 없어 농촌을 깨우치고 상하이 훙커우 공원에서 목숨을 건 의열 투쟁을 결단한 것입니다! "
            "조국의 자유를 향한 뜨거운 투혼에 대해 무엇이든 더 들려드리겠습니다!"
        )
        kw = "윤봉길"

    return text, kw, False, False


def get_persona_answer_with_keyword(char_name, user_question, char_data):
    """
    LLM 답변을 생성하며, 대화 맥락에서 우측 패널에 띄울 '가장 핵심적인 시각 자료 검색어(명사 1개)'를 분리 추출.
    외부 API 장애 시에도 지능형 아카이브 폴백(Graceful Degradation)을 가동하여 무중단 서비스 제공.
    반환: (clean_response_text, extracted_keyword, is_llm: bool, is_out_of_scope: bool)
    """
    api_keys = get_gemini_api_keys()
    
    # 범위 외(Out-of-Scope) 질문 여부 사전 스크리닝
    out_of_scope_keywords = [
        "날씨", "비 오", "눈 오", "기온", "점심", "저녁", "아침", "메뉴", "밥", "식사", "배고파", "맛집", "국밥",
        "코딩", "파이썬", "자바", "프로그래밍", "코드", "버그", "주식", "코인", "비트코인",
        "아이돌", "연예인", "노래", "영화", "드라마", "게임", "축구", "야구"
    ]
    is_out_pre = any(w in user_question.lower() for w in out_of_scope_keywords)

    # 1. API 키가 전혀 없는 경우 즉시 지능형 아카이브 폴백 가동
    if not api_keys:
        return generate_contextual_persona_fallback(char_name, user_question, char_data)

    system_instruction = (
        f"당신은 대한민국의 역사 인물 '{char_name}'입니다. {char_data['persona']}\n"
        f"다음은 당신의 삶에 대한 역사적 사실입니다: {char_data['fact']}\n"
        f"답변 시 역사적 사실에 기반하여 성실히 답하세요.\n\n"
        f"【Out-of-Scope(범위 외) 질문 방어 규칙】\n"
        f"1. 사용자가 역사와 무관한 질문(예: 날씨, 오늘 점심 메뉴, 코딩/프로그래밍, 주식, 최신 연예인 소식, 현대 잡담 등)을 던지면, 현대의 잡담에 휩쓸리지 말고 반드시 '{char_name}'의 인물 페르소나, 어투, 가치관을 철저히 유지한 채 재치 있게 역사적 교훈과 조국의 의미로 연결하여 정중히 거절하세요.\n"
        f"2. 예시 (안중근 의사 페르소나):\n"
        f"   - 유저: '오늘 날씨 어때?'\n"
        f"   - 답변: '하늘의 날씨를 살피는 것도 중요하지만, 우리 민족의 마음에 드리운 일제의 그림자를 거두는 것이 더 급선무요. 날씨에 마음을 쓰기보다 오늘의 역사를 돌아보시오.'\n"
        f"3. 마크다운 기호(**, * 등) 문법 오류가 나지 않도록 일반 텍스트 중심으로 자연스럽고 정갈하게 출력하세요.\n\n"
        f"【필수 시각자료 지침】\n"
        f"답변의 가장 마지막 줄에, 답변 내용 중 학생들에게 화면으로 보여줄 '가장 핵심적인 역사적 장소/유물/사건 명사 1개'를 반드시 다음 형식으로 1줄만 작성하세요:\n"
        f"[시각자료_검색어: 키워드]\n"
        f"- 예시 1: 답변에서 탑골공원을 언급한 경우 -> [시각자료_검색어: 탑골공원]\n"
        f"- 예시 2: 답변에서 하얼빈역을 언급한 경우 -> [시각자료_검색어: 하얼빈역]\n"
        f"- 예시 3: 답변에서 서대문형무소를 언급한 경우 -> [시각자료_검색어: 서대문형무소]\n"
        f"- 범위 외 질문 방어 답변이거나 일상적 인사 등으로 특정 시각 자료를 특정하기 어려울 경우 -> [시각자료_검색어: 없음]"
    )

    candidate_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-1.5-flash",
        "gemini-pro"
    ]

    total_keys = len(api_keys)
    last_error = ""

    # 🔄 [Key Failover 루프]
    for key_idx, api_key in enumerate(api_keys, start=1):
        try:
            genai.configure(api_key=api_key)
        except Exception as config_err:
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
                    target_kw = None
                    kw_match = re.search(r"\[시각자료_검색어:\s*([^\]]+)\]", raw_text)
                    if kw_match:
                        extracted = kw_match.group(1).strip()
                        if extracted not in STOP_KEYWORDS and len(extracted) >= 2:
                            target_kw = extracted
                        clean_text = re.sub(r"\[시각자료_검색어:\s*[^\]]+\]", "", raw_text).strip()
                    else:
                        clean_text = raw_text.strip()

                    is_out_scope = is_out_pre or (target_kw == "없음")
                    return clean_text, target_kw, True, is_out_scope
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                if any(q in err_str.lower() for q in ["429", "quota", "resourceexhausted", "rate limit", "ratelimit"]):
                    key_hit_quota = True
                    break
                continue

        if key_hit_quota and key_idx < total_keys:
            continue

    # 외부 API가 모두 실패(키 오류, 쿼터 소진, 네트워크 타임아웃)하더라도 앱이 크래시되지 않도록
    # 지능형 역사 아카이브 폴백을 즉각 가동하여 100% 무중단 응답 보장
    return generate_contextual_persona_fallback(char_name, user_question, char_data)

# [5. 대화 세션 및 UI 렌더링 동기화]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("is_cached") is True:
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">⚡ 초고속 즉시 답변 (캐시 적용됨)</span></div>',
                unsafe_allow_html=True
            )
        elif message.get("is_saved") is True:
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">🧠 실시간 AI 생성 완료 (DB 저장됨)</span></div>',
                unsafe_allow_html=True
            )
        elif message.get("role") == "assistant":
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 500;">📜 역사 사료 아카이브 응답</span></div>',
                unsafe_allow_html=True
            )
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

# [6. 학습자용 원클릭 추천 질문 (Tag / Chip UI)]
st.markdown("---")
st.markdown("💡 **추천 질문** *(클릭 시 바로 질문)*")
preset_cols = st.columns(3)
selected_preset = None

current_presets = PRESET_QUESTIONS.get(char_name, [])
for idx, q_txt in enumerate(current_presets):
    with preset_cols[idx]:
        if st.button(q_txt, key=f"chip_{char_name}_{idx}", use_container_width=True):
            selected_preset = q_txt

# 질문 입력창 및 프리셋 트리거 통합
chat_input_val = st.chat_input(f"{char_name} 님께 질문할 내용을 입력하세요...")
active_prompt = chat_input_val or selected_preset or st.session_state.pop("pending_prompt", None)

if active_prompt:
    # 1. 사용자 질문 렌더링 & 세션 저장
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # 2. 어시스턴트 메시지 (Cache-Aside 패턴 + 단계별 로딩 피드백 + 실시간 스트리밍)
    with st.chat_message("assistant"):
        with st.status(f"📜 {char_name} 님의 사료 검증 및 답변 처리 중...", expanded=True) as status_box:
            # 1. Cache Check: DB에 동일/유사 질의가 이미 존재하는지 먼저 조회
            st.write("🔍 Q&A 캐시 데이터베이스 조회 중...")
            cached_res = get_cached_qa(char_name, active_prompt)
            
            if cached_res:
                response_text, target_keyword, hit_count, match_type = cached_res
                is_cached = True
                type_desc = "완전 일치" if match_type == "exact" else "유사 질문 자동 식별"
                st.write(f"⚡ **[Cache HIT: {type_desc}]** 기존 데이터베이스에서 답변을 즉시 호출했습니다. (누적 활용: {hit_count}회, API 지연시간 0초)")
            else:
                is_cached = False
                st.write("1️⃣ **[Cache MISS]** 외부 LLM API(Gemini) 호출 및 페르소나 답변 생성 중...")
                response_text, target_keyword, is_llm, is_out_of_scope = get_persona_answer_with_keyword(char_name, active_prompt, char_data)
                
                # [핵심 가드레일] 오직 실제 Gemini LLM이 성공적으로 추론한 정상 답변만 DB 캐시에 저장 (임시 폴백 및 Out-of-Scope 제외)
                if is_llm and not is_out_of_scope:
                    is_saved = save_cached_qa(char_name, active_prompt, response_text, target_keyword, is_llm=True, is_out_of_scope=False)
                    if is_saved:
                        st.write("💾 신규 고품질 문답 쌍을 Q&A 캐시 DB에 안전하게 저장했습니다.")
                else:
                    is_saved = False
                    st.write("ℹ️ 안전한 사료 기반 아카이브 응답 모드로 처리되었습니다. (캐시 무결성을 위해 저장은 생략됩니다)")
            
            st.write("2️⃣ 답변 맥락 기반 공인 역사 사료(Wikimedia) 탐색 중...")
            wiki_image = search_wikimedia_image_strict(target_keyword) if target_keyword else None
            
            full_response = f"{response_text}\n\n🔗 [박물관 오픈아카이브 원문]({char_data['url']})"
            if is_cached:
                status_box.update(label=f"⚡ {char_name} 님의 답변 즉시 로드 완료! (Cache HIT)", state="complete", expanded=False)
            else:
                status_box.update(label=f"✅ {char_name} 님의 답변 준비 완료!", state="complete", expanded=False)

        # 텍스트 스트리밍 타이핑 효과로 생동감 제공
        def text_stream_generator(text):
            words = text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.015)

        if hasattr(st, "write_stream"):
            st.write_stream(text_stream_generator(full_response))
        else:
            st.markdown(full_response)

        # Q&A 캐싱 상태 뱃지 및 토스트 피드백
        if is_cached:
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">⚡ 초고속 즉시 답변 (캐시 적용됨)</span></div>',
                unsafe_allow_html=True
            )
            st.toast("⚡ 초고속 캐시 적중! 외부 API 호출 없이 로컬 DB에서 즉시 답변을 반환했습니다.", icon="⚡")
        elif is_saved:
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">🧠 실시간 AI 생성 완료 (DB 저장됨)</span></div>',
                unsafe_allow_html=True
            )
            st.toast("🧠 실시간 AI 생성 완료! 새로운 질문-답변이 캐시 DB에 저장되었습니다.", icon="🧠")
        else:
            st.markdown(
                '<div style="margin: 6px 0;"><span style="background-color: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 500;">📜 역사 사료 아카이브 응답</span></div>',
                unsafe_allow_html=True
            )
            st.toast("📜 역사 사료 기반의 안정적인 페르소나 응답이 생성되었습니다.", icon="📜")

        # 사료 이미지 또는 폴백 렌더링
        msg_record = {"role": "assistant", "content": full_response, "is_cached": is_cached, "is_saved": is_saved}
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

    # 세션에 최종 저장하여 새로고침 시에도 유지
    st.session_state.messages.append(msg_record)
