
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chat_service import (
    ask_cybersecurity_assistant,
)


router = APIRouter(
    prefix="/api/chat",
    tags=["AI Chat"],
)


class ChatRequest(BaseModel):

    question: str = Field(
        min_length=5,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


@router.post("/ask")
def ask_question(request: ChatRequest):

    try:

        result = ask_cybersecurity_assistant(
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "success": True,
            "question": request.question,
            "data": result,
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print("Chat error:", str(error))

        raise HTTPException(
            status_code=502,
            detail="AI response generation failed.",
        )
