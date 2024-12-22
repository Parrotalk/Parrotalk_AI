import logging
from dotenv import load_dotenv
import openai
import json
import os
from openai import OpenAI

# OpenAI API 키 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 템플릿 문자열 정의
template_string = """ 
작업: 대화 상대방의 발화 기록과 다음 문장을 참고하여 당신이 고객이나 친구 입장이 되어, 자연스럽고 상황에 적합한 답변을 3개 생성해라. 
문장: {sentence} 
대화 기록: {total_combined_text}

[조건]
1. 상대방의 말투(존댓말/반말)에 따라 답변 말투를 일치시켜라.
2. 예/아니오로 답변할 수 있는 질문에는 다양한 응답 옵션을 제시하라.
3. '안녕하세요'와 같은 기본적인 인사말만 포함된 문장은 답변하지 않고 빈칸을 반환하라.
4. 상대방의 질문이나 요청을 이해하고 이에 적절한 답변을 하도록 하라.
5. 답변은 간결하면서도 대화의 맥락을 고려하여 의미 있게 작성하라.

[답변 형식]
추천 문장 1: 첫번째 문장
추천 문장 2: 두번째 문장
추천 문장 3: 세번째 문장
"""
def generate_sentence(total_combined_text, sentence):
    # 템플릿 문자열을 상황에 맞게 완성
    prompt = template_string.format(
        total_combined_text=total_combined_text,
        sentence=sentence
    )

    try:
        client = OpenAI(
            api_key=openai.api_key,
        )

        # OpenAI ChatGPT API를 호출하여 응답 받기
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Response in json format"},
                {"role": "user", "content": prompt}
            ],
            # response_format 지정하기
            response_format = {"type":"json_object"}
        )

        # 응답 메시지를 추출
        customer_response = response.choices[0].message.content

        # JSON 형식으로 파싱
        output_dict = json.loads(customer_response)

        return output_dict

    except Exception as e:
        # 오류가 발생할 경우 로깅
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return None
    
#print(generate_sentence('오늘 날씨 춥더라'))