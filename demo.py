import csv
import json

# Load JSON data
data = '''
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day", "Happy day", "Hi", "What's up?", "how are you"],
      "responses": ["Hello, thanks for asking", "Hey!", "Good to see you again", "Welcome!", "Hello I am good, how can I help?"],
      "context": [""]
    },
    {
      "tag": "new_help_desk_ticket",
      "patterns": ["How can I submit a new help desk ticket?", "How do I go about creating a new help desk ticket?", "What is the process for submitting a fresh help desk ticket?", "Can you guide me on how to initiate a new help desk ticket?", "In what way can I formally request assistance through a help desk ticket?", "What steps do I need to follow to generate a new help desk ticket?", "Could you provide instructions on how to start a new help desk ticket?", "What is the procedure for logging a new help desk ticket?", "Is there a specific way I should go about opening a help desk ticket?", "Can you share the steps involved in submitting a help desk ticket for assistance?", "How can I officially raise a request by creating a help desk ticket?"],
      "responses": ["Through help desk Mail Id or help desk portal"],
      "context": [""]
    },
    {
      "tag": "mail_id",
      "patterns": ["Through help desk Mail Id or help desk portal", "What is the email address for the help desk?", "Could you provide the mail ID for the help desk?", "What is the email contact for the help desk?", "How can I reach the help desk via email?", "What email should I use to contact the help desk?", "Can you share the help desk's email ID with me?", "How do I find the mail ID for the help desk?", "What is the designated email address for help desk inquiries?", "May I have the email information for the help desk?", "Could you let me know the email ID to contact the help desk?"],
      "responses": ["helpdesk@srmtech.com"],
      "context": [""]
    },
    {
      "tag": "track_status",
      "patterns": ["How to track my ticket status?", "What is the process for monitoring the status of my ticket?", "Can you guide me on checking the progress of my ticket?", "How do I keep tabs on the status of my submitted ticket?", "Is there a way to stay informed about the current status of my ticket?", "What steps should I take to track the status of my ticket?"],
      "responses": ["Once the ticket has been created you can either login to help desk portal or simply click on 'view ticket' from the mail you have received"],
      "context": [""]
    },
    {
      "tag": "existing_ticket",
      "patterns": ["How do I check the status of my existing ticket?", "Can you guide me on how to verify the status of my existing ticket?", "How can I find out the current status of the ticket I've already submitted?", "Is there a way to check the progress of my existing ticket?"],
      "responses": ["Can see in help desk portal"],
      "context": [""]
    },
    {
      "tag": "required_information",
      "patterns": ["What information should I include when submitting a help desk ticket?", "What details are required when I submit a help desk ticket?", "Can you guide me on the necessary information for a help desk ticket submission?", "What specific information should be included when creating a help desk ticket?", "Are there particular details I need to provide when submitting a help desk request?", "What information is essential for a complete help desk ticket submission?"],
      "responses": ["BU, phone, subject, description of your issue, priority level, classification"],
      "context": [""]
    },
    {
      "tag": "resolution_time",
      "patterns": ["When will my ticket be solved?", "How long does it typically take for help desk ticket to be resolved?", "What is the expected resolution time for my ticket?", "Can you provide an estimate of when my ticket will be resolved?", "When can I anticipate the resolution of my submitted ticket?", "Is there a timeframe for resolving my ticket issue?", "How soon can I expect my ticket to be addressed and resolved?"],
      "responses": ["Depends on priority levels."],
      "context": [""]
    },
    {
      "tag": "attachments_screenshots",
      "patterns": ["Can I add attachments or screenshots to my help desk ticket?", "Is there an option to include attachments or screenshots when creating a help desk ticket?", "Can I upload files or screenshots along with my help desk ticket submission?", "Is it possible to attach additional documents or screenshots to my help desk ticket?", "Are there provisions for including attachments or screen captures with my help desk request?", "Can I supplement my help desk ticket with attachments or screenshots for more context?"],
      "responses": ["Yes, you can. In help desk portal there is an attachment option, there u can add your attachments nd screenshots"],
      "context": [""]
    },
    {
      "tag": "update_existing_ticket",
      "patterns": ["How can I provide additional information or updates to an existing ticket?", "What is the procedure for adding extra details or updates to an already submitted ticket?", "Can you guide me on how to include additional information or updates to an existing ticket?", "Is there a way to supplement an existing ticket with extra information or updates?", "How can I provide further details or updates to a ticket that I've already submitted?", "What steps should I take to append additional information or updates to an existing ticket?"],
      "responses": ["Through help desk portal, in that you can update your existing ticket. In view option you can see multiple sub-options of your raised tickets"],
      "context": [""]
    },
    {
      "tag": "feedback",
      "patterns": ["How can I provide feedback the support I received for my help desk ticket?", "What is the process for giving feedback on the support received for my help desk ticket?", "How can I share my feedback regarding the assistance received for my help desk ticket?", "Can you guide me on offering feedback for the help I received with my help desk ticket?"],
      "responses": ["In the help desk portal you can see the feedback option in that you can provide your feedback."],
      "context": [""]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "I am leaving", "Have a Good Day", "Till next time"],
      "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
      "context": [""]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
      "responses": ["Happy to help!", "Any time!", "My pleasure"],
      "context": [""]
    },
    {
      "tag": "noanswer",
      "patterns": [""],
      "responses": ["Sorry, can't understand you", "please provide more context or details so that I can better understand and help you", "Not sure I understand"],
      "context": [""]
    }
  ]
}
'''

# Parse JSON
intents = json.loads(data)

# CSV file path
csv_file_path = 'intents_data.csv'

# Writing CSV
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    # Creating a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Writing header
    csvwriter.writerow(['tag', 'patterns', 'responses', 'context'])

    # Writing data rows
    for intent in intents['intents']:
        csvwriter.writerow([intent['tag'], '| '.join(intent['patterns']), '| '.join(intent['responses']), '| '.join(intent['context'])])

print(f'CSV file has been created successfully at {csv_file_path}')