import streamlit as st
from openai import OpenAI

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

# [2. UI 설정]
st.set_page_config(page_title="타임톡(Time-Talk)", page_icon="📜")
st.title("📜 타임톡(Time-Talk)")
st.caption("대한민국역사박물관 오픈아카이브 데이터 기반 AI 페르소나 챗봇")

# 사이드바 인물 선택
char_name = st.sidebar.selectbox("대화할 인물을 선택하세요:", list(knowledge_base.keys()))
char_data = knowledge_base[char_name]

# 메인 UI 출력
col1, col2 = st.columns([1, 2])
with col1:
    try:
        st.image(char_data["img"], use_container_width=True)
    except:
        st.warning("이미지 파일을 확인하세요.")
with col2:
    st.write(f"### {char_name} 의사/열사")
    st.info(f"**학습 페르소나:** {char_data['persona']}")

# RAG생성
def get_persona_answer(char_name, user_question, char_data):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    
    # 1. 시스템 설정 (인물의 페르소나 주입)
    system_prompt = f"당신은 {char_name}입니다. {char_data['persona']}"
    
    # 2. 지식 주입 (우리가 가진 사실 데이터)
    context = f"다음은 당신의 삶에 대한 역사적 사실입니다: {char_data['fact']}"
    
    # 3. LLM에게 질문 (OpenAI API 호출 예시)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{system_prompt}\n{context}\n답변 시 출처 정보를 하단에 반드시 제공하세요."},
                {"role": "user", "content": user_question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[{char_name}의 답변] : API 연결 오류가 발생했습니다.: {str(e)}"
        
# [3. 대화 로직 및 RAG 기능]
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("역사에 대해 궁금한 점을 질문해보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 답변 생성 (실제 API 연동 시 OpenAI 호출부)
    with st.chat_message("assistant"):
        
        # 실제 답변 생성
        response_text = get_persona_answer(char_name, prompt, char_data)
        
        full_response = response_text + f"\n\n🔗 [근거 자료 확인하기]({char_data['url']})"
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
