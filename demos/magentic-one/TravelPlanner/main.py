import asyncio
import os
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from utils import format_response, get_city_prompt

# Add Phi-4 imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Azure OpenAI client for Magentic One
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

# Initialize Phi-4 client
phi4_client = None
phi4_model_name = "microsoft/Phi-4-multimodal-instruct"  # Using the exact correct model name
phi4_token = os.environ.get("GITHUB_TOKEN")
print(f"GitHub Token loaded: {'Yes, token found' if phi4_token else 'No'}", file=sys.stderr)

# Function to verify if Phi-4 is accessible
def verify_phi4_connection():
    global phi4_client, phi4_model_name
    try:
        if not phi4_token:
            print(f"GitHub token not found in environment variables", file=sys.stderr)
            return False
        
        # Create the client
        test_client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(phi4_token),
        )
        
        # Test the connection with a short prompt
        test_response = test_client.complete(
            messages=[UserMessage(content="Hi")],
            max_tokens=5,
            model=phi4_model_name
        )
        
        # If we got here without an exception, the connection works
        print(f"✓ Phi-4 connection verified", file=sys.stderr)
        print(f"✓ Response: {test_response.choices[0].message.content}", file=sys.stderr)
        
        # Set the global client
        phi4_client = test_client
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error connecting to Phi-4: {str(e)}", file=sys.stderr)
        return False

# Try to verify the connection
PHI4_AVAILABLE = verify_phi4_connection()
print(f"Phi-4 available: {PHI4_AVAILABLE}", file=sys.stderr)

# List of supported cities
CITIES = ["London", "Paris", "Rome", "Barcelona", "Amsterdam"]

# Maintain interview session state
interview_sessions = {}

@app.route('/')
def index():
    return render_template('index.html', cities=CITIES)

@app.route('/plan', methods=['POST'])
def plan_trip():
    try:
        data = request.get_json()
        city = data.get('city', 'London')
        days = data.get('days', 3)
        interests = data.get('interests', '')
        
        # Create prompt for the agent
        prompt = get_city_prompt(city, days, interests)
        
        # Log the request to terminal
        print(f"\n=== NEW TRIP PLANNING REQUEST ===", file=sys.stderr)
        print(f"City: {city}", file=sys.stderr)
        print(f"Days: {days}", file=sys.stderr)
        print(f"Interests: {interests}", file=sys.stderr)
        print(f"Prompt:\n{prompt}\n", file=sys.stderr)
        
        # Run the async function in a synchronous context
        result = asyncio.run(run_travel_agent_team(prompt))
        
        # Format the response
        formatted_result = format_response(result)
        
        return jsonify({"status": "success", "response": formatted_result})
    except Exception as e:
        error_msg = str(e)
        print(f"\n=== ERROR GENERATING TRAVEL PLAN ===", file=sys.stderr)
        print(f"Error: {error_msg}\n", file=sys.stderr)
        return jsonify({"status": "error", "message": error_msg})

async def run_travel_agent_team(prompt):
    """
    Run a team of specialized Magentic One agents to create a travel plan.
    This follows the pattern from the full Magentic One architecture documentation.
    """
    print(f"\n=== INITIALIZING MAGENTIC ONE TEAM ===", file=sys.stderr)
    try:
        # Initialize the model client with Azure OpenAI
        model_client = AzureOpenAIChatCompletionClient(
            model=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT"),
            azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT", "gpt-4o"),
            api_version=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_API_VERSION", "2024-08-01-preview"),
            azure_ad_token_provider=token_provider,
        )
        print(f"Azure OpenAI client initialized", file=sys.stderr)
        
        # Create the lead Orchestrator agent responsible for planning and coordination
        orchestrator = AssistantAgent(
            "Orchestrator",
            system_message="""You are the Orchestrator agent responsible for task decomposition, planning, and coordination in a Magentic One multi-agent team.
Your responsibilities include:
1. Creating a detailed plan to tackle the travel planning task
2. Maintaining a Task Ledger with facts and educated guesses
3. Delegating subtasks to specialized agents (TravelResearcher, TravelPlanner, ComputerTerminal)
4. Tracking progress in a Progress Ledger and checking task completion status
5. Revising the plan if progress is not being made

You will work with:
- TravelResearcher: A web browsing agent that can search for travel information
- TravelPlanner: A coder agent that can organize and format travel plans
- ComputerTerminal: An agent that can execute code if needed

Begin by analyzing the travel request, creating a plan, and delegating subtasks to the appropriate agents.
""",
            model_client=model_client,
        )
        print(f"Orchestrator agent created", file=sys.stderr)
        
        # Create a web surfer agent to search for travel information - using headless mode
        web_surfer = MultimodalWebSurfer(
            "TravelResearcher",
            model_client=model_client,
            headless=True,  # Set to True to avoid opening browser windows
            animate_actions=False,  # Disable animations for better performance
        )
        print(f"Web surfer agent created (headless mode)", file=sys.stderr)
        
        # Create a coder agent to organize and format the travel plan
        planner = MagenticOneCoderAgent(
            "TravelPlanner",
            model_client=model_client,
        )
        print(f"Travel planner agent created", file=sys.stderr)
        
        # Create a terminal agent to execute any necessary code
        terminal = CodeExecutorAgent(
            "ComputerTerminal",
            code_executor=LocalCommandLineCodeExecutor(),
        )
        print(f"Terminal agent created", file=sys.stderr)
        
        # Create a team with the Orchestrator as the lead agent
        # The order is important - Orchestrator must be first
        team = MagenticOneGroupChat(
            [orchestrator, web_surfer, planner, terminal],
            model_client=model_client,
        )
        print(f"Agent team created with Orchestrator leading", file=sys.stderr)
        
        print(f"\n=== RUNNING MAGENTIC ONE TEAM WITH ORCHESTRATOR ===", file=sys.stderr)
        print(f"This may take 3-8 minutes as the team browses travel websites and coordinates...", file=sys.stderr)
        print(f"The Orchestrator will plan and coordinate work (browser windows hidden):", file=sys.stderr)
        
        # Capture the content from the streaming events
        final_content = ""
        last_update_time = asyncio.get_event_loop().time()
        timeout_seconds = 300  # 5-minute timeout for the entire process
        
        # Process streaming events to show agent activity in terminal
        async for event in team.run_stream(task=prompt):
            # Update the last activity timestamp
            last_update_time = asyncio.get_event_loop().time()
            
            # Print agent activities in the terminal
            if hasattr(event, 'sender'):
                sender = getattr(event, 'sender', 'Unknown')
                print(f"\n[AGENT: {sender}]", file=sys.stderr)
            
            if hasattr(event, 'thought') and event.thought:
                print(f"THINKING: {event.thought[:300]}...", file=sys.stderr)
            
            if hasattr(event, 'content') and event.content:
                content = event.content
                
                # Handle case where content is a list (contains text and images)
                if isinstance(content, list):
                    # Extract the text part from the list if available
                    text_content = ""
                    for item in content:
                        if isinstance(item, str):
                            text_content += item
                            print(f"CONTENT (LIST-TEXT): {item[:150]}...", file=sys.stderr)
                        else:
                            print(f"CONTENT (LIST-NON-TEXT): [Image or other non-text content]", file=sys.stderr)
                    
                    # Only update if we got meaningful text
                    if len(text_content) > 50 and not text_content.strip().startswith("We are working to address"):
                        final_content = text_content
                else:
                    # Handle normal string content
                    print(f"CONTENT: {content[:150]}...", file=sys.stderr)
                    
                    # Only update final_content if it's not a repeated "working on" message
                    if len(content) > 50 and not content.strip().startswith("We are working to address"):
                        final_content = content  # Keep the latest substantial content
            
            if hasattr(event, 'action'):
                action = event.action
                if hasattr(action, 'url') and action.url:
                    print(f"BROWSING: {action.url}", file=sys.stderr)
                elif hasattr(action, 'name'):
                    print(f"ACTION: {action.name}", file=sys.stderr)
            
            # Print separator for readability
            print("-" * 50, file=sys.stderr)
            
            # Check for timeout - stop if no progress is being made
            current_time = asyncio.get_event_loop().time()
            if current_time - last_update_time > timeout_seconds:
                print(f"\n=== TIMEOUT: NO PROGRESS FOR {timeout_seconds} SECONDS ===", file=sys.stderr)
                break
        
        print(f"\n=== MAGENTIC ONE TEAM COMPLETED ===", file=sys.stderr)
        
        # If we didn't get a proper result, provide a fallback
        if not final_content or final_content.strip().startswith("We are working to address"):
            print(f"\n=== GENERATING FALLBACK TRAVEL PLAN ===", file=sys.stderr)
            # Extract city, days from prompt
            city_name = "London"  # Default
            days_count = 1  # Default
            
            for line in prompt.split('\n'):
                if "trip to" in line:
                    parts = line.split("trip to ")
                    if len(parts) > 1:
                        city_parts = parts[1].split(" for ")
                        if len(city_parts) > 0:
                            city_name = city_parts[0].strip()
                if "for " in line and " days" in line:
                    parts = line.split("for ")
                    for part in parts:
                        if " days" in part:
                            days_text = part.split(" days")[0].strip()
                            try:
                                days_count = int(days_text)
                            except:
                                pass
            
            fallback_plan = f"""
# {city_name} Travel Plan ({days_count} Day{'s' if days_count > 1 else ''})

I apologize, but I was unable to gather the most current information about {city_name} due to web access limitations. Here's a general travel plan based on typical attractions and experiences:

## Day 1: Essential {city_name}

### Morning
- Start your day at a local café for breakfast
- Visit the main museum or historical site
- Explore the central area on foot

### Afternoon
- Enjoy lunch at a recommended local restaurant
- Visit another key attraction
- Take time to shop or relax in a park

### Evening
- Dinner at a popular restaurant
- Optional evening entertainment or relaxation at your accommodation

## Transportation Tips
- Public transport is typically the best option for getting around
- Consider a day pass for unlimited travel
- Walking is often the best way to experience the city center

## Budget Considerations
- Accommodation: $150-300 per night
- Meals: $50-100 per day
- Attractions: $50-100 per day
- Transportation: $20-40 per day

*For the most current information about attractions, events, and restaurants, I recommend checking the official tourism website or contacting your accommodation once you arrive.*
"""
            final_content = fallback_plan
        
        # Close the web surfer browser
        await web_surfer.close()
        
        # Close the model client
        await model_client.close()
        
        # Return the final travel plan
        return final_content
    except Exception as e:
        print(f"\n=== ERROR IN MAGENTIC ONE TEAM ===", file=sys.stderr)
        print(f"Error details: {str(e)}", file=sys.stderr)
        
        # If we encounter an error, return a basic plan
        return f"""
# Travel Plan Error

I apologize, but I encountered an error while creating your travel plan: {str(e)}

Here are some general tips for planning your trip:
- Check official tourism websites for the most current information
- Consider booking accommodations in advance
- Research public transportation options for getting around
- Look for local events happening during your visit

Please try again later or with a different request.
"""

async def run_interview_agent(session_id, city=None, days=None, user_input=None):
    """
    Run a general purpose interview agent to engage users in a conversation about their travel interests.
    This agent uses Phi-4-mini for efficient interview processing.
    """
    print(f"\n=== INTERVIEW AGENT SESSION {session_id} ===", file=sys.stderr)
    try:
        # Initialize session if new
        if session_id not in interview_sessions:
            interview_sessions[session_id] = {
                "city": city,
                "days": days,
                "interests": [],
                "conversation": [],
                "complete": False,
                "stage": 0  # Track interview progress: 0=start, 1=asking categories, 2=specific interests, 3=finalizing
            }
        
        session = interview_sessions[session_id]
        
        # Update session with latest info
        if city:
            session["city"] = city
        if days:
            session["days"] = days
        
        # Create system message for the interview agent based on current stage
        system_message = """You are an expert travel consultant interviewing a traveler to discover their interests.
Your goal is to ask focused questions to identify what types of experiences they'd enjoy on their trip.
Be conversational, friendly and concise. Ask one question at a time.
Don't overwhelm the user with too many options at once."""
        
        # Add stage-specific instructions
        if session["stage"] == 0:
            system_message += """
Begin by asking what general category of interests they have (e.g., food, culture, history, nature, nightlife, family activities).
After they respond, move to stage 1 for category follow-up questions."""
        elif session["stage"] == 1:
            system_message += """
Based on their general categories, ask about specific interests within those categories.
For example, if they mentioned food, ask about cuisine preferences, fine dining vs street food, food tours, etc.
After they respond, move to stage 2 for more specific questions."""
        elif session["stage"] == 2:
            system_message += """
Now that you have some specific interests, ask follow-up questions to refine their preferences.
For example, if they like museums, ask which types (art, history, science) they prefer.
When you have enough detail, move to stage 3 for summarization."""
        elif session["stage"] == 3:
            system_message += """
Summarize the interests you've gathered and ask if there's anything else they'd like to add.
If they're satisfied, mark the interview as complete."""
        
        # Add the user input to the conversation history
        if user_input:
            session["conversation"].append({"role": "user", "content": user_input})
        
        # If this is the first message, add a greeting without calling the model
        if not session["conversation"]:
            if session["city"] and session["days"]:
                starter_prompt = f"I'm planning your {session['days']}-day trip to {session['city']}. I'd like to learn more about your travel interests to create the perfect itinerary for you. What types of activities or experiences are you generally interested in when traveling?"
            else:
                starter_prompt = "I'd like to learn more about your travel interests to create the perfect itinerary for you. What types of activities or experiences are you generally interested in when traveling?"
                
            session["conversation"].append({"role": "assistant", "content": starter_prompt})
            
            return {
                "response": starter_prompt,
                "interests": session["interests"],
                "complete": session["complete"]
            }
            
        # Prepare messages for the model
        phi4_messages = []
        
        # Add system message
        if phi4_client:
            phi4_messages.append(SystemMessage(content=system_message))
        
        # Add conversation history
        for msg in session["conversation"]:
            if msg["role"] == "user":
                phi4_messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                phi4_messages.append(AssistantMessage(content=msg["content"]))
        
        # Get response based on available client
        if phi4_client:
            # Use Phi-4 client
            print(f"Using Phi-4-mini for interview agent", file=sys.stderr)
            response = phi4_client.complete(
                messages=phi4_messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=300,
                model=phi4_model_name
            )
            agent_response = response.choices[0].message.content
        else:
            # Fallback to Azure OpenAI
            print(f"Phi-4 not available, falling back to Azure OpenAI", file=sys.stderr)
            # Initialize the model client with Azure OpenAI
            model_client = AzureOpenAIChatCompletionClient(
                model=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT", "gpt-4o"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT"),
                azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT", "gpt-4o"),
                api_version=os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_API_VERSION", "2024-08-01-preview"),
                azure_ad_token_provider=token_provider,
            )
            
            # Convert to autogen format
            autogen_messages = []
            for msg in session["conversation"]:
                autogen_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Call Azure OpenAI
            response = await model_client.create(messages=autogen_messages)
            agent_response = response.choices[0].message.content
            await model_client.close()
        
        # Update the conversation
        session["conversation"].append({"role": "assistant", "content": agent_response})
        
        # Extract interests from the conversation
        if user_input:
            # Extract potential interests from user input
            session = extract_interests_from_conversation(session)
            
        # Check if we should advance to the next stage
        session = progress_interview_stage(session, agent_response)
        
        # Return the response and session state
        return {
            "response": agent_response,
            "interests": session["interests"],
            "complete": session["complete"]
        }
    except Exception as e:
        print(f"\n=== ERROR IN INTERVIEW AGENT ===", file=sys.stderr)
        print(f"Error details: {str(e)}", file=sys.stderr)
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}. Let's start over with your interests. What would you like to experience on your trip?",
            "interests": [],
            "complete": False
        }

def extract_interests_from_conversation(session):
    """
    Extract interest keywords from the conversation history.
    Uses simple keyword extraction for now, but could be enhanced with NLP techniques.
    """
    # Common travel interest keywords to look for
    interest_categories = [
        "museum", "art", "history", "architecture", "food", "restaurant", "dining", "cafe", "coffee", 
        "shopping", "market", "park", "garden", "nature", "hiking", "beach", "nightlife", "bar", 
        "music", "concert", "theater", "show", "tour", "local", "authentic", "landmark", "sightseeing",
        "adventure", "sport", "family", "kid", "children", "luxury", "budget", "photography", "religious",
        "temple", "church", "cathedral", "castle", "palace", "festival", "event", "wine", "beer",
        "culture", "local", "traditional", "modern", "scenic", "view", "boat", "cruise", "walking"
    ]
    
    # Get the last user message
    last_user_message = ""
    for msg in reversed(session["conversation"]):
        if msg["role"] == "user":
            last_user_message = msg["content"].lower()
            break
    
    # Extract interests from the user's message
    for word in interest_categories:
        if word in last_user_message and word not in session["interests"]:
            session["interests"].append(word)
    
    return session

def progress_interview_stage(session, agent_response):
    """
    Determine if the interview should progress to the next stage.
    """
    # Check if the agent is asking a new type of question indicating stage progression
    if session["stage"] == 0 and len(session["interests"]) >= 1:
        # User has shared initial interests, move to specific questions
        session["stage"] = 1
    elif session["stage"] == 1 and len(session["interests"]) >= 3:
        # User has shared specific interests, move to refinement
        session["stage"] = 2
    elif session["stage"] == 2 and len(session["interests"]) >= 5:
        # User has shared enough refined interests, move to summary
        session["stage"] = 3
    elif session["stage"] == 3:
        # Check if the agent's response includes a summary
        summary_indicators = ["summary", "summarize", "based on our conversation", "you've mentioned", 
                             "your interests include", "to recap", "you're interested in"]
        
        if any(indicator in agent_response.lower() for indicator in summary_indicators):
            # Agent has provided a summary, mark as complete
            session["complete"] = True
    
    return session

@app.route('/interview/start', methods=['POST'])
def start_interview():
    """
    Start a new interest interview session.
    """
    try:
        data = request.get_json()
        city = data.get('city')
        days = data.get('days')
        
        # Generate a unique session ID
        session_id = os.urandom(8).hex()
        
        # Initialize the interview agent and get the first response
        result = asyncio.run(run_interview_agent(session_id, city, days))
        
        # Add the model name to the response
        model_name = "Microsoft Phi-4-mini" if phi4_client else "Azure OpenAI GPT-4o"
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "response": result["response"],
            "interests": result["interests"],
            "complete": result["complete"],
            "model_name": model_name
        })
    except Exception as e:
        error_msg = str(e)
        print(f"\n=== ERROR STARTING INTERVIEW ===", file=sys.stderr)
        print(f"Error: {error_msg}\n", file=sys.stderr)
        return jsonify({"status": "error", "message": error_msg})

@app.route('/interview/continue', methods=['POST'])
def continue_interview():
    """
    Continue an existing interview session.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_input = data.get('user_input')
        
        if not session_id or session_id not in interview_sessions:
            return jsonify({"status": "error", "message": "Invalid or expired session"})
        
        # Continue the interview with the user's input
        result = asyncio.run(run_interview_agent(session_id, user_input=user_input))
        
        # Add the model name to the response
        model_name = "Microsoft Phi-4-mini" if phi4_client else "Azure OpenAI GPT-4o"
        
        return jsonify({
            "status": "success",
            "response": result["response"],
            "interests": result["interests"],
            "complete": result["complete"],
            "model_name": model_name
        })
    except Exception as e:
        error_msg = str(e)
        print(f"\n=== ERROR CONTINUING INTERVIEW ===", file=sys.stderr)
        print(f"Error: {error_msg}\n", file=sys.stderr)
        return jsonify({"status": "error", "message": error_msg})

if __name__ == '__main__':
    print("\n=== STARTING CITY TRIP PLANNER APPLICATION ===", file=sys.stderr)
    print(f"Server will be available at http://127.0.0.1:5000", file=sys.stderr)
    print(f"Expected processing time: 3-8 minutes per request", file=sys.stderr)
    print(f"Environment settings:", file=sys.stderr)
    print(f"AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT: {os.environ.get('AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT', 'Not set')}", file=sys.stderr)
    print(f"AZURE_OPENAI_CHAT_COMPLETION_API_VERSION: {os.environ.get('AZURE_OPENAI_CHAT_COMPLETION_API_VERSION', 'Not set')}", file=sys.stderr)
    print(f"AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT: {os.environ.get('AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT', 'Not set')}", file=sys.stderr)
    app.run(debug=True)