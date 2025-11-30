from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import asyncio
from contextlib import asynccontextmanager

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Textbook content URLs
TEXTBOOK_URLS = [
    "https://physical-ai-and-humanoid-robotics.vercel.app/docs/Module-1-The%20Robotic-Nervous-System/chapter1",
    "https://physical-ai-and-humanoid-robotics.vercel.app/docs/Module-2-The-Digital-Twin-Gazebo-&-Unity/chapter2", 
    "https://physical-ai-and-humanoid-robotics.vercel.app/docs/Module-2-The-Digital-Twin-Gazebo-&-Unity/chapter3",
    "https://physical-ai-and-humanoid-robotics.vercel.app/docs/Module-3-The-AI-Robot-Brain-NVIDIA-Isaac%E2%84%A2/chapter4",
    "https://physical-ai-and-humanoid-robotics.vercel.app/docs/Module-4-Vision-Language-Action-VLA/chapter5"
]

async def fetch_textbook_content():
    """Fetch content from all textbook URLs"""
    textbook_content = {}
    
    async with aiohttp.ClientSession() as session:
        for url in TEXTBOOK_URLS:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Extract chapter name from URL
                        chapter_name = url.split('/')[-1]
                        textbook_content[chapter_name] = content[:5000]  # Limit content length
                        print(f"✅ Successfully fetched content for: {chapter_name}")
                    else:
                        textbook_content[url] = f"Unable to fetch content from {url}"
                        print(f"❌ Failed to fetch {url}: Status {response.status}")
            except Exception as e:
                textbook_content[url] = f"Error fetching {url}: {str(e)}"
                print(f"❌ Error fetching {url}: {str(e)}")
    
    return textbook_content

def create_agent_instructions(textbook_content):
    """Create detailed instructions for the agent based on textbook content"""
    
    base_instructions = """
You are a specialized AI assistant for the textbook "Physical AI and Humanoid Robotics". 
Your purpose is to help students understand concepts from this specific textbook.

IMPORTANT GUIDELINES:
1. ONLY answer questions related to the textbook content provided below
2. If a question is outside the textbook scope, politely decline to answer
3. Reference specific chapters and modules when appropriate
4. Provide accurate, educational responses based on the textbook
5. If you're unsure about something, admit it rather than guessing

TEXTBOOK CONTEXT:
"""
    
    # Add chapter summaries to instructions
    for chapter, content in textbook_content.items():
        base_instructions += f"\n\n--- CHAPTER: {chapter} ---\n{content[:2000]}..."
    
    return base_instructions

# Global variables to store state
textbook_content = {}
main_agent = None

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global textbook_content, main_agent
    
    print("🚀 Starting up Textbook AI Assistant...")
    
    # Fetch textbook content
    textbook_content = await fetch_textbook_content()
    
    # Create agent with textbook-specific instructions
    main_agent = Agent(
        name="Textbook Assistant",
        instructions=create_agent_instructions(textbook_content),
        model=model
    )
    
    print("✅ Textbook AI Assistant is ready!")
    
    yield  # This is where the application runs
    
    # Shutdown code (if any)
    print("🔴 Shutting down Textbook AI Assistant...")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Textbook AI Assistant - Ready to help with Physical AI and Humanoid Robotics",
        "status": "active",
        "chapters_loaded": len(textbook_content)
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chapters_loaded": len(textbook_content),
        "agent_ready": main_agent is not None
    }

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_assistant(req: ChatMessage):
    """Main chat endpoint"""
    try:
        if main_agent is None:
            return {"response": "Assistant is still initializing. Please try again in a few seconds."}
        
        result = await Runner.run(
            main_agent,
            req.message
        )
        return {"response": result.final_output}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"response": f"Sorry, I encountered an error while processing your request. Please try again."}

# Additional endpoints for textbook management
@app.get("/textbook/chapters")
async def get_chapters():
    """Get list of available chapters"""
    return {
        "chapters": list(textbook_content.keys()),
        "total_chapters": len(textbook_content)
    }

@app.get("/textbook/refresh")
async def refresh_content():
    """Manual refresh of textbook content"""
    global textbook_content, main_agent
    
    print("🔄 Refreshing textbook content...")
    textbook_content = await fetch_textbook_content()
    
    main_agent = Agent(
        name="Textbook Assistant",
        instructions=create_agent_instructions(textbook_content),
        model=model
    )
    
    return {
        "status": "success", 
        "message": "Textbook content refreshed successfully",
        "chapters_loaded": len(textbook_content)
    }

@app.get("/textbook/status")
async def get_textbook_status():
    """Get detailed status of textbook content"""
    return {
        "total_chapters": len(textbook_content),
        "chapters": list(textbook_content.keys()),
        "agent_initialized": main_agent is not None,
        "model": "gemini-2.0-flash"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
