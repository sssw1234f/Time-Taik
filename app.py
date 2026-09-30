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
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    return api_key

# [2. UI 설정]
st.set_page_config(page_title="타임톡(Time-Talk)", page_icon="📜")
st.title("📜 타임톡(Time-Talk)")
st.caption("대한민국역사박물관 오픈아카이브 데이터 기반 AI 페르소나 챗봇")

# 사이드바 인물 선택 (불필요한 모델 선택 UI 완전 제거)
char_name = st.sidebar.selectbox("대화할 인물을 선택하세요:", list(knowledge_base.keys()))
char_data = knowledge_base[char_name]

# -------- Teacher worksheet generation ----------
st.sidebar.markdown("---")
if st.sidebar.button("교사용 수업 활동지 생성하기"):
    msgs = st.session_state.get("messages", [])
    def generate_worksheet(msgs, char_name, char_data):
        md = f"# {char_name} 학습 활동지\n\n"
        md += "## Ⅰ. 토론 주제\n"
        md += f"- 초등학생: {char_name}은 언제, 어디서 활약했나요?\n"
        md += f"- 중학생: {char_name}의 행동이 당시 한국에 어떤 영향을 미쳤나요?\n"
        md += f"- 고등학생: {char_name}의 사상을 현대에 어떻게 적용할 수 있나요?\n\n"
        md += "## Ⅱ. 빈칸 채우기 퀴즈\n"
        md += f"1. {char_name}은 ___년 ___월 ___일에 ___에서 활동을 시작했다.\n"
        md += f"2. {char_name}이 남긴 주요 업적은 ___이다.\n"
        md += f"3. {char_name}이 사용하는 말투는 \"{char_data['persona']}\"이다.\n\n"
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

# RAG & 제미나이 답변 생성 함수 (쿼터 초과 시 자동 폴백 지원)
def get_persona_answer(char_name, user_question, char_data):
    current_key = get_gemini_api_key()
    if not current_key:
        return f"[{char_name}의 답변] : Gemini API 키가 설정되지 않았습니다. 환경변수 또는 Streamlit Secrets에 GEMINI_API_KEY(또는 GOOGLE_API_KEY)를 등록해주세요."

    try:
        genai.configure(api_key=current_key)
        
        # 시스템 프롬프트 (인물의 페르소나 및 사실 데이터 주입)
        system_instruction = (
            f"당신은 {char_name}입니다. {char_data['persona']}\n"
            f"다음은 당신의 삶에 대한 역사적 사실입니다: {char_data['fact']}\n"
            f"답변 시 출처 정보를 하단에 반드시 제공하세요."
        )
        
        # 429 Quota(할당량) 초과 방지를 위한 순차 대체 모델 리스트
        candidate_models = [
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
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
                
                if hasattr(response, "text") and response.text:
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
            except Exception as e:
                last_error = str(e)
                # 429(할당량 초과) 또는 404 발생 시 다음 가용 모델로 즉시 자동 시도
                continue
                
        # 모든 모델 후보가 소진되었을 때 친절한 안내 메시지 출력
        if "429" in last_error or "quota" in last_error.lower():
            return f"[{char_name}의 답변] : 구글 무료 계정의 요청 할당량(Quota)이 일시적으로 초과되었습니다. 약 20~30초 후에 다시 질문을 입력해 주세요."
        return f"[{char_name}의 답변] : API 연결 오류가 발생했습니다: {last_error}"
    except Exception as e:
        return f"[{char_name}의 답변] : 시스템 오류가 발생했습니다: {str(e)}"
        
# [3. 대화 로직 및 RAG 기능]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력 및 응답 생성
if prompt := st.chat_input("역사에 대해 궁금한 점을 질문해보세요!"):
    # 1. 사용자 질문 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 답변 생성 (스피너로 응답 대기 표시)
    with st.chat_message("assistant"):
        with st.spinner(f"{char_name} 님이 답변을 작성하고 있습니다..."):
            response_text = get_persona_answer(char_name, prompt, char_data)
            full_response = response_text + f"\n\n🔗 [근거 자료 확인하기]({char_data['url']})"
            st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
