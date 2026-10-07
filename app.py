import streamlit as st
import os
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

# [API 키 취득 헬퍼]
def get_gemini_api_key():
    # 1. 수동 입력 세션 키 우선
    if st.session_state.get("user_gemini_key"):
        return st.session_state["user_gemini_key"].strip()

    # 2. 환경변수 탐색
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()

    # 3. Streamlit secrets 탐색
    try:
        if hasattr(st, "secrets"):
            for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"]:
                if k in st.secrets and st.secrets[k]:
                    return str(st.secrets[k]).strip()
    except Exception:
        pass
    return None

# [2. UI 설정]
st.set_page_config(page_title="타임톡(Time-Talk)", page_icon="📜")
st.title("📜 타임톡(Time-Talk)")
st.caption("대한민국역사박물관 오픈아카이브 데이터 기반 AI 페르소나 챗봇")

# 사이드바 설정
st.sidebar.header("⚙️ 대화 설정")
char_name = st.sidebar.selectbox("대화할 인물을 선택하세요:", list(knowledge_base.keys()))
char_data = knowledge_base[char_name]

# API 키 연결 상태 및 수동 입력 폴백
current_key = get_gemini_api_key()
if current_key:
    st.sidebar.success("🟢 Gemini API 연결됨")
else:
    st.sidebar.error("🔴 Gemini API 키 미등록")
    user_key_input = st.sidebar.text_input(
        "API 키 직접 입력:",
        type="password",
        placeholder="AIzaSy...",
        help="Streamlit Secrets에 등록되지 않은 경우 여기에 바로 입력하세요."
    )
    if user_key_input:
        st.session_state["user_gemini_key"] = user_key_input
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

# RAG & 제미나이 답변 생성 함수
def get_persona_answer(char_name, user_question, char_data):
    api_key = get_gemini_api_key()
    if not api_key:
        return f"[{char_name}의 답변] : Gemini API 키가 설정되지 않았습니다. 왼쪽 사이드바에 API 키를 직접 입력하거나 Streamlit Secrets에 등록해주세요."

    try:
        genai.configure(api_key=api_key)
        
        system_instruction = (
            f"당신은 {char_name}입니다. {char_data['persona']}\n"
            f"다음은 당신의 삶에 대한 역사적 사실입니다: {char_data['fact']}\n"
            f"답변 시 역사적 사실에 기반하여 성실히 답하세요."
        )
        
        # 신속하고 안정적인 실 서비스 모델 리스트
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-3.8-flash",
            "gemini-flash-latest"
        ]
        
        last_error = ""
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(user_question)
                
                text_content = ""
                if hasattr(response, "text") and response.text:
                    text_content = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    text_content = response.candidates[0].content.parts[0].text
                
                if text_content and text_content.strip():
                    return text_content.strip()
            except Exception as e:
                last_error = str(e)
                continue
                
        if "429" in last_error or "quota" in last_error.lower():
            return f"[{char_name}의 답변] : 구글 무료 계정의 요청 할당량(Quota)이 일시적으로 초과되었습니다. 약 20~30초 후에 다시 질문을 입력해 주세요."
        return f"[{char_name}의 답변] : API 오류 발생: {last_error}"
    except Exception as e:
        return f"[{char_name}의 답변] : 시스템 오류: {str(e)}"

# [3. 대화 세션 및 렌더링 (Streamlit 표준 placeholder 패턴)]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력 및 응답 생성
if prompt := st.chat_input("역사에 대해 궁금한 점을 질문해보세요!"):
    # 1. 사용자 질문 렌더링 & 세션 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 메시지 (placeholder 활용으로 확실한 렌더링 보장)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(f"{char_name} 님이 답변을 작성하고 있습니다..."):
            response_text = get_persona_answer(char_name, prompt, char_data)
            full_response = f"{response_text}\n\n🔗 [근거 자료 확인하기]({char_data['url']})"
        
        # placeholder에 최종 텍스트 주입 (증발 방지)
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
