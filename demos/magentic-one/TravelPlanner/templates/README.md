# City Trip Planner

A modern web application for planning city trips using Microsoft's Magentic One framework for intelligent, web-aware travel recommendations and Phi-4 for personalized travel interviews.

![City Trip Planner](https://via.placeholder.com/800x400?text=City+Trip+Planner)

## Features

- **Interactive Travel Planning**: Generate personalized travel plans for popular cities
- **AI-Powered Web Research**: Utilizes Magentic One agents to browse the web for the latest information
- **Guided Travel Interviews**: Uses Microsoft Phi-4 model to conduct personalized travel interest interviews
- **Modern UI**: Sleek, glassy interface with a dark theme for comfortable viewing
- **Multiple Interest Input Methods**: Choose between manual interest input or guided AI interviews
- **Multi-day Itineraries**: Create detailed plans spanning multiple days based on your preferences

## Supported Cities

- London
- Paris
- Rome
- Barcelona
- Amsterdam

## Technology Stack

- **Backend**: Python with Flask web framework
- **Frontend**: HTML, CSS, JavaScript
- **AI Models**:
  - Microsoft Phi-4-multimodal-instruct for personalized interviews
  - Magentic One framework with GPT-4o for travel planning and web research
- **Authentication**: Azure AD for Azure OpenAI services

## Setup

### Prerequisites

- Python 3.9+
- GitHub Personal Access Token with `models:read` permission (for Phi-4)
- Azure account with OpenAI service access (for Magentic One)

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# GitHub Token for Phi-4
GITHUB_TOKEN=your_github_token

# Azure OpenAI Service
AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT=gpt-4o
AZURE_OPENAI_CHAT_COMPLETION_API_VERSION=2024-08-01-preview
```

### Installation

1. Clone the repository
2. Create and activate a virtual environment:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Windows PowerShell
source .venv/bin/activate     # On Linux/MacOS
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
python main.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Select a city from the dropdown menu
2. Choose the number of days for your trip
3. Share your interests either:
   - By typing them manually in the text area, or
   - Through a guided interview with the Phi-4 AI assistant
4. Click "Generate Trip Plan" to create your personalized itinerary
5. Review the generated plan with attractions, activities, and practical tips

## Components

### Main Application (`main.py`)

The core application file that handles:
- Flask web server setup
- AI model initialization (Phi-4 and Azure OpenAI)
- API endpoints for trip planning and interviews
- Magentic One agent orchestration

### Utility Functions (`utils.py`)

Helper functions for:
- Response formatting
- City-specific prompt generation
- Text processing

### Web Interface (`templates/index.html`)

Modern, responsive interface featuring:
- City and duration selection
- Interest collection (manual or AI-guided)
- Real-time interview feedback
- Plan display with markdown rendering

## Extending the Application

### Adding New Cities

To add support for additional cities, update the `CITIES` list in `main.py` and add corresponding prompts in `utils.py`.

### Customizing the UI

The application uses inline CSS for styling. Modify the styles in `templates/index.html` to customize the appearance.

### Using Alternative Models

The application is designed to gracefully fall back to Azure OpenAI if Phi-4 is unavailable. To use a different model for interviews, modify the `verify_phi4_connection` function in `main.py`.

## License

MIT

## Acknowledgments

- Microsoft Magentic One framework for intelligent agent orchestration
- Microsoft Phi-4 model for conversational capabilities
- Flask community for the lightweight web framework