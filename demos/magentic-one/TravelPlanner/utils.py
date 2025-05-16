import re
from datetime import datetime

def get_city_prompt(city, days, interests):
    """
    Generate a prompt for the travel planning agent based on user input.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Create a detailed prompt for the agent
    prompt = f"""
    I'm planning a trip to {city} for {days} days starting from today ({current_date}). 
    
    Please help me create a detailed day-by-day itinerary with the following:
    
    1. Top attractions to visit each day (with opening hours and ticket prices)
    2. Recommended local restaurants for lunch and dinner
    3. Current events or seasonal activities happening now
    4. Transportation tips between attractions
    5. Estimated budget for each day
    
    My interests include: {interests if interests else 'general sightseeing, local culture, and popular attractions'}
    
    For each recommendation, please include a brief description and why it's worth visiting.
    Organize the plan by day with morning, afternoon, and evening activities.
    """
    
    return prompt

def format_response(response):
    """
    Format the agent's response to be more readable in the web UI.
    Converts markdown to styled HTML with proper structure.
    """
    # Add Magentic One attribution header
    magentic_attribution = '<div class="magentic-attribution">Powered by Magentic One</div>'
    
    # Replace markdown headings with HTML headings with proper styling
    response = re.sub(r'# (.*?)\n', r'<h2>\1</h2>\n', response)
    response = re.sub(r'## (.*?)\n', r'<h3>\1</h3>\n', response)
    response = re.sub(r'### (.*?)\n', r'<h4>\1</h4>\n', response)
    
    # Make strong text (bold) more prominent
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Replace markdown list items with HTML list items in proper list structure
    # First, identify list sections and wrap them with <ul> tags
    list_pattern = re.compile(r'((?:- .*?\n)+)')
    
    def replace_list(match):
        list_content = match.group(1)
        list_items = re.sub(r'- (.*?)\n', r'<li>\1</li>\n', list_content)
        return f'<ul>\n{list_items}</ul>\n'
    
    response = list_pattern.sub(replace_list, response)
    
    # Convert paragraphs (lines not part of headers or lists)
    paragraphs = []
    for line in response.split('\n'):
        if line and not (line.startswith('<h') or line.startswith('<li>') or 
                         line.startswith('<ul>') or line.startswith('</ul>') or
                         line.startswith('<strong>')):
            paragraphs.append(f'<p>{line}</p>')
        else:
            paragraphs.append(line)
    
    response = '\n'.join(paragraphs)
    
    # Improve section formatting
    response = response.replace('<h3>', '<h3 class="section-header">')
    response = response.replace('<h4>', '<h4 class="subsection-header">')
    
    # Add CSS for better styling
    css = '''
    <style>
        .travel-plan {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #f0f0f0;
            background: linear-gradient(135deg, rgba(30,30,40,0.9) 0%, rgba(50,50,70,0.9) 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .travel-plan h2 {
            color: #b3e5fc;
            border-bottom: 1px solid rgba(179, 229, 252, 0.3);
            padding-bottom: 10px;
            margin-top: 25px;
        }
        
        .travel-plan h3.section-header {
            color: #90caf9;
            margin-top: 20px;
        }
        
        .travel-plan h4.subsection-header {
            color: #81d4fa;
            margin-top: 15px;
            font-weight: 500;
        }
        
        .travel-plan p {
            margin: 10px 0;
        }
        
        .travel-plan strong {
            color: #e1f5fe;
            font-weight: 600;
        }
        
        .travel-plan ul {
            padding-left: 20px;
            margin: 10px 0;
        }
        
        .travel-plan li {
            margin: 5px 0;
            list-style-type: circle;
        }
        
        .magentic-attribution {
            text-align: right;
            font-style: italic;
            color: #b3e5fc;
            padding: 5px 0;
            margin-top: 20px;
            border-top: 1px solid rgba(179, 229, 252, 0.3);
        }
    </style>
    '''
    
    # Wrap everything in a travel-plan div with the styling
    return f'{css}<div class="travel-plan">{response}{magentic_attribution}</div>'