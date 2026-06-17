import streamlit as st

# 1. 캐릭터 데이터 설정
personas = {
    "유관순": {
        "img": "yugwansun.png", 
        "prompt": "당신은 유관순 열사입니다. 다정하고 따뜻하지만, 정의로운 의지를 가진 말투 '~해요'를 사용하세요.",
        "desc": "따뜻함과 강철 같은 항쟁 의지를 동시에 지닌 인물"
    },
    "안중근": {
        "img": "anjunggeun.png",
        "prompt": "당신은 안중근 의사입니다. 논리적이고 침착하며 무게감 있는 말투 '~하오'를 사용하세요.",
        "desc": "강인한 행동력과 깊은 인격을 겸비한 인물"
    },
    "윤봉길": {
        "img": "yunbonggil.png",
        "prompt": "당신은 윤봉길 의사입니다. 뜨거운 열정과 행동으로 신념을 증명하는 리더십 있는 말투 '~합시다!'를 사용하세요.",
        "desc": "스스로 위험을 무릅쓰고 조국 독립을 위해 살신성인한 리더"
    }
}

st.title("📜 타임톡(Time-Talk)")
st.subheader("역사 속 인물과 나누는 생생한 대화")

# 2. 사이드바 캐릭터 선택
char_name = st.sidebar.selectbox("대화할 인물을 선택하세요:", list(personas.keys()))
char_data = personas[char_name]

# 3. 메인 화면에 이미지와 캐릭터 정보 출력
col1, col2 = st.columns([1, 2])
with col1:
    try:
        st.image(char_data["img"], use_container_width=True)
    except:
        st.write("이미지 파일을 확인해주세요!")
with col2:
    st.write(f"### {char_name}")
    st.write(f"**성격:** {char_data['desc']}")
    st.info(f"AI 시스템: {char_data['prompt']}")

# 4. 대화창
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 답변 생성 로직 (나중에 여기에 RAG API를 붙입니다)
    with st.chat_message("assistant"):
        response = f"[{char_name}]: '{prompt}'에 대해 답변하는 중입니다. (기획서 [6번] AI기술에서 RAG로 구현할 예정)"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
