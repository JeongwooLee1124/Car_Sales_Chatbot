import streamlit as st
import time
from PIL import Image
from chatgpt_llm import graph_chain
import asyncio

def image_to_base64(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 챗봇 아바타 이미지 불러오기
chatbot_avatar = Image.open("./images/salesguy.jpg")
chatbot_avatar_base64 = image_to_base64(chatbot_avatar)

# 초기 질문용 아바타 이미지 불러오기
chatbot_initial_avatar = Image.open("./images/logo.png")
chatbot_initial_avatar_base64 = image_to_base64(chatbot_initial_avatar)

async def typing_effect(message_placeholder):
    typing_text = "에이전트가 답변을 작성 중입니다"
    while st.session_state.awaiting_response:
        st.session_state.typing_index = (st.session_state.typing_index + 1) % 6
        message_placeholder.markdown(
            f"<div class='assistant-message'>"
            f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
            f"<div><div class='assistant-name'>{st.session_state.agent}</div>"
            f"<div class='chatbox typing'>{typing_text}{'.' * st.session_state.typing_index}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        await asyncio.sleep(0.3)

async def fetch_answer_and_typing(inputs, message_placeholder):
    loop = asyncio.get_event_loop()
    future_result = loop.run_in_executor(None, graph_chain.invoke, inputs)
    
    typing_task = asyncio.create_task(typing_effect(message_placeholder))

    result = await future_result
    typing_task.cancel()

    try:
        await typing_task
    except asyncio.CancelledError:
        pass

    final_answer = result.get("generation", "Sorry, I couldn't find an answer.")
    st.session_state.final_answer = final_answer

    # 최종 답변을 placeholder에 스트리밍
    message_text = ""
    for char in final_answer:
        message_text += char
        message_placeholder.markdown(
            f"<div class='assistant-message'>"
            f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
            f"<div><div class='assistant-name'>{st.session_state.agent}</div>"
            f"<div class='chatbox'>{message_text}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        await asyncio.sleep(0.02)  # 여기서 타이핑 속도를 조정할 수 있음

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    if "typing_index" not in st.session_state:
        st.session_state.typing_index = 0

    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""

    if "initial_choice_made" not in st.session_state:
        st.session_state.initial_choice_made = False
        st.session_state.agent = None

    st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stChatMessageUser {
            display: none !important;
        }
        .chatbox {
            border: none;
            border-radius: 15px;
            padding: 10px;
            display: inline-block;
            word-wrap: break-word;
            max-width: 80%;
            margin-bottom: 10px;
            background-color: #f0f0f0;
        }
        .initial-question {
            background-color: #ffffff;
            border: none;
            text-align: left;
            float: left;
            clear: both;
            border-radius: 15px;
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
        }
        .typing {
            min-width: 250px;
            min-height: 40px;
            line-height: 1.5;
        }
        .user-message {
            background-color: #d1e7dd;
            border: none;
            text-align: right;
            float: right;
            clear: both;
            border-radius: 15px;
        }
        .assistant-message {
            background-color: #ffffff;
            border: none;
            text-align: left;
            float: left;
            clear: both;
            border-radius: 15px;
            display: flex;
            align-items: center;
        }
        .assistant-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .assistant-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .chat-container {
            margin: 0 auto;
            max-width: 800px;
        }
        .choice-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            font-size: 18px;
            text-align: center;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"""
            <div class="assistant-message">
                <img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>
                <div>
                    <div class="assistant-name">{st.session_state.agent}</div>
                    <div class="chatbox">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chatbox user-message'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.initial_choice_made:
        initial_question = """안녕하세요, 언제나 최상의 차량 정보를 제공하는 에피카입니다. 저희와 함께해 주셔서 감사합니다. 차량에 대해 궁금한 점이나 도움이 필요하시면 언제든지 말씀해 주세요. 오늘은 어떤 정보가 필요하신가요?"""
        # st.session_state.messages.append({"role": "assistant", "content": initial_question})
        st.markdown(f"""
        <div class="assistant-message initial-question">
            <img src='data:image/png;base64,{chatbot_initial_avatar_base64}' class='assistant-avatar'>
            <div>
                <div class="assistant-name">EPIKAR</div>
                <div class="chatbox">{initial_question}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("정확한 차량의 스펙과 성능 정보를 알려주세요.", key="specs"):
            choice = "정확한 차량의 스펙과 성능 정보"
            agent = '인포마스터'
        elif st.button("자동차 딜러의 의견과 이야기가 듣고 싶어요.", key="opinions"):
            choice = "딜러의 의견과 이야기"
            agent = '코넥토'
        elif st.button("저를 위한 자동차 구매 혜택과 특별 프로모션도 궁금해요", key="promotions"):
            choice = "나를 위한 구매 혜택과 특별 프로모션"
            agent = '게인지니'
        else:
            choice = None
            agent = None

        if choice:
            st.session_state.initial_choice_made = True
            st.session_state.choice = choice
            st.session_state.agent = agent

            follow_up_message = (f"안녕하세요, 저는 피카지니 입니다. 고객님께 최고의 딜을 제공해드리기 위해 여기 있습니다. \n"
                                 "고객님의 나이와 차량 구매 목적을 말씀해주시면 더 나은 추천을 드릴 수 있습니다. \n "
                                 "또한, 현대, 기아, 르노 중에서 관심 있는 브랜드가 있으면 알려주세요.")
            st.session_state.messages.append({"role": "assistant", "content": follow_up_message})

            st.rerun()

    if st.session_state.initial_choice_made:
        prompt = st.chat_input("메세지를 입력해 주세요:")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f"<div class='chatbox user-message'>{prompt}</div>", unsafe_allow_html=True)

            st.session_state.awaiting_response = True
            st.session_state.prompt = prompt

            message_placeholder = st.empty()

            asyncio.run(fetch_answer_and_typing({"question": prompt, 'chat_history': st.session_state.messages, 'agent': st.session_state.agent}, message_placeholder))

            message_placeholder.markdown(
                f"<div class='assistant-message'>"
                f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
                f"<div><div class='assistant-name'>{st.session_state.agent}</div>"
                f"<div class='chatbox'>{st.session_state.final_answer}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.final_answer})
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    main()
