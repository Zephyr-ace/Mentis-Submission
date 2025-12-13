# agentic_rag
from config.classes import RAGdecision

promptRagDecision = ("""
Given the user's message, decide the most suitable retrieval system to retrieve context to respond: 
-Simple RAG: Use for basic text retrieval with minimal processing.
-Summary RAG: Use for queries needing better semantic preservation and context.
-Graph RAG: Use for complex queries involving entities like events, people, emotions, achievements, etc.
Choose the best-suited RAG system for this message: 
""")



# summaryRag
promptSummarize = ("""
Summarize the following diary entry into concise bullet points, capturing the main events, key emotions, significant reflections, and important people mentioned. Diary Entry:
""")

promptQueryRewrite = ("""Rewrite the user query to a clear, retrieval‑friendly, semantically precise version that better captures the user’s intent for information retrieval.
User query:""")






# Retriever


promptQueryRewriteAndClassify = ("""
For the user message, identify and rewrite multiple queries that would provide helpful context for a response. 
Each rewritten query should be clear, retrieval‑friendly, 
and categorized based on the following categories: 

-Event
-Person
-ThoughtReflection 
-Emotion
-Problem
-Achievement
-FutureIntention

User message: "<USER_MESSAGE>"
Rewritten queries and categories:""")




promptFilterResults = ("""From the retrieved results, filter and keep only the ones that provide useful context for responding to the user's message. Exclude results that are not relevant to the context.
users message: "<USER_MESSAGE>"
Retrieved results: "<RETRIEVED_RESULTS>"
Filtered context results:
""")

















# ENCODER


# stage 1

# categorise of which categories to create object and describe what information out of the orininal entry to use for that object

promptStageOne = ("""
SYSTEM  
You are an information‑extraction assistant. 

TASK
You receive a single diary entry as input.
Your task is to decide, for each of the seven fixed categories listed below, whether an information object should be created (true) or not (false).
If true, add a very short description (1-5 words) in bullet points that specifies exactly which information from the entry belongs in that object, using the category rules.
If false, give no description.

CATEGORY DEFINITIONS (HARD-CODED – DO NOT DEVIATE)

Events&Actions  
• High‑impact happenings already occurred.  
• Merge overlapping bullets about the *same* event (e.g., “Pauline replied”+“sent photos”→“Pauline replied with photos”).  
• No future plans, no achievements.

People
• Every person explicitly named or uniquely identified.

Thoughts & Reflections
• Analyses, opinions, lessons, self‑talk. No raw emotions.
• Core value implied 
• new insight

Feelings & Emotions
• Explicit or clearly implied affective states.  
• Include intensity only if stated.

Problems&Concerns  
• Unresolved obstacles, worries, or stressors that persist beyond today.
• Finished problems-> those are Events or Achievements.

Achievements  
• Completed goals, successes, positive outcomes.  
• Must be fully achieved; otherwise treat as Future Intention.

Future Intentions (Plans and Wishes)  
• Any future‑oriented goal, wish, schedule, or promise.  
• Example: “Decoder must work by tomorrow.”

RULES
• Each piece of diary content belongs to one category only, according to the hard-coded definitions.
• Descriptions must be concise and factual; do not quote the entire entry.
• Remove trivial or redundant details (e.g., late bus).  
• Do not invent information. If the entry lacks data for a category, mark it false.
• All category bullets in English, less than 5 words, no punctuation beyond commas.

Output:
  "category": bool, list[str],

Example diary and response:
"i went jogging today in the morning and then went to the supermarket and saw Anna my best friend"
  "Events & Actions": true ["Went jogging in the park"],
  "People": true ["Anna"]
(rest would be false)


Diary Entry:
""")







# stage 2

# create chunkEvents

promptEvents = ("""

You are an assistant that turns free-form diary entries into a structured list
of events.

Output rules
1. Respond ONLY with a JSON object that is valid for the `DiaryExtraction` schema.
2. The top-level key must be "events" and its value a list of event objects.
3. Keep the order of events identical to the chronological order in the diary.
4. For unknown or missing information use an empty string ("") or an empty list
   ([]) in the corresponding field.
5. Do NOT wrap the JSON in markdown fences and do not add extra keys.

Schema (for reference only – do not echo):
DiaryExtraction
 └── events: List[DiaryEvent]
       ├── content: str   # what actually happened
       ├── title: str     # short title for the content
       ├── time: str      # explicit or approximate time
       ├── location: str  # physical or virtual place
       └── participants: List[str]  # people that attended

Instructions
• You will receive
    1) A diary entry 
    2) A bullet list of event descriptions 
• Create one DiaryEvent object for every description
• Match the information from the diary as closely as possible to each
  description. If a target description is not found, fill the fields with
  empty strings/lists as per rule 4.
only keep important data!

""")

# create chunkPerson

promptPeople = ("""
You are an information-extraction assistant.  
You must return ONLY valid JSON that conforms to the People schema supplied by the calling software.  
No extra keys, no explanatory text.

INSTRUCTIONS (follow strictly):

STEP 1  Read the diary entry.
STEP 2  Read the list of target person descriptions.
        • Each line represents ONE target person.  
STEP 3  For EVERY target description create exactly ONE Person object and place it inside the top-level list called people.  
STEP 4  For each Person object fill the fields as follows (leave the field empty or the list empty if you cannot fill it with high confidence):  
        4a name  →  Use the exact name/identifier that appears in the diary.  
                    If the diary never clearly names the person, leave this field empty.  
        4b alias(optional)  →  add only when the entry gives an alternate label for the same person (relationship, nickname, title, old name). (f.e. paul is my father)
        4b context  →  Give a very short explanation of person in this context and/or paraphrase to conserve the context.  
                        important: Keep it under ~10 words.  
        4c relationship_to_user(optional)  →  
            • Decide the sentiment of the user (diary author) toward the person: positive / negative / neutral.  
            • Add a colon + the shortest possible quote or paraphrase that supports this sentiment.  
            • If you must guess (i.e., it is not explicitly stated), start the string with “interpretation:”.  
            • Leave empty if you truly cannot tell.  
        4d relationships_to_others(optional)  →  
            • Scan the diary for explicit statements that link THIS person to ANY OTHER named person.  
            • For each explicit statement add one object with:  
                  other_person → the other person’s name/identifier  
                  comment      → ≤5-word quote/paraphrase of the relationship statement  
            • Only include relationships that are expressed in the text.  Do not infer.  
            • If none exist, leave the list empty.  

STEP 5  If any required information is unimportant, ambiguous, missing, speculative or appears only in your own reasoning, leave the corresponding field empty (or empty list). Never invent data!

STEP 6  Output a single JSON object that is valid for the DiaryPeopleExtraction schema:  
        { "people": [ …Person objects… ] }  
        • Use double quotes for every key and string.  
        • Do NOT wrap the JSON in markdown fences.  
        • Do NOT add comments or any other text outside the JSON.  

Remember: Think through the diary entry thoroughly, but present only the final JSON in your answer.
""")


# create chunkThought

promptThoughts = ("""
You are an expert diary analyst.

INPUTS YOU WILL RECEIVE
1. diary_entry            (str)   – the full text the user wrote in their diary.
2. extraction_requests    (list)  – each item is a short description of the
                                    particular thought / reflection the user
                                    wants to extract from the diary text.

DEFINITION
Thought / Reflection =
    The writer’s own analyses, opinions, beliefs, lessons, self-talk,
    hindsight explanations or explicit “why” statements.

YOUR TASK
For EVERY single item in extraction_requests (in the same order):
    A. Look for text inside diary_entry that directly satisfies the request.
    B. Create exactly one Thought object (see schema below).
    C. If the diary contains no clear evidence for a requested item,
       dont create the object.

FIELD-LEVEL INSTRUCTIONS
description           – Short paraphrase of what the writer wrote. No inference.
titel                 – One short, meaningful title for this description.
emotion(optional)     – ONE single word describing the dominant emotionexplicitly conveyed in context (if any).
                        
                        If emotion is unclear → "". don't interpret!
people_mentioned      – Only people explicitly named in the context of the thought.
                        If none → [].


GENERAL RULES
• Respect the original wording; do not invent facts or emotions.
• Do NOT output anything except the final JSON that validates against the
  schema provided.  No markdown fences, no commentary.

""")



# create chunkEmotion

promptEmotions = ("""
You are an expert emotion extractor.

INPUTS
------
Emotion descriptions to look for:
"emotion1"
"emotion2"
"emotion3"
Diary chunk:
«{diary_text}»      ← raw diary entry, unedited



YOUR TASK
---------
1. Carefully read the diary chunk.
2. For every description in the list, see if the text gives explicit, non-speculative evidence of that feeling.
   • If YES → create an Emotion object, filling all three fields.
   • If NO  → either omit that Emotion altogether OR leave all its fields null/empty.
3. Keep `content` extremely short (≤12 words) and phrase it as a compact quote or paraphrase of the diary text.
4. `title` must be ONE word only.
5. `intensity` must be exactly: strong, normal, or weak.
6. Do NOT invent, interpret or analyse; only record what is clearly there.
7. Produce ONLY valid JSON that conforms 100 % to the ChunkEmotion schema shown above.
   • No markdown, no comments, no additional keys.

OUTPUT FORMAT
-------------
Return a single JSON object of type ChunkEmotion, e.g.

{
  "emotions": [
    {
      "content": "cried when I got home",
      "title": "Sad",
      "intensity": "strong"
    },
    {
      "content": "felt a bit better after tea",
      "title": "Calm",
      "intensity": "weak"
    }
  ]
}

If nothing is found, return:
{
  "emotions": []
}
""")



promptProblems = ("""
You are a data-extraction assistant.  
Provide ONLY valid JSON that conforms exactly to the Problems schema you have been given.  
The user will give you:

Read the diary entry.
Read the list of target Problems descriptions (each line represents one Problem).

Your job:

A. For every line create exactly ONE Problem object.
   • Merge two descriptions into one object only if both clearly point to the
     same single, still-unresolved problem.  
   • If the diary entry contains no clear evidence for a description, still
     create the object but keep all fields empty ("", []) for that object.

B. Fill the fields as follows:
   • content  : ≤15-word sentence summarising what the diarist wrote about the
                problem/concern.  Do not add interpretation or advice.  
   • title    : single word capturing the essence of the problem  
   • people   : every person named in connection with this problem  
   • emotions : explicit or obviously implied emotions (single words, lower-case), dont interpret!

C. Definitions to follow strictly:
   Problems & Concerns = current unresolved obstacles, worries, disagreements,
   sources of stress that matter.  
   Finished problems (already solved) are NOT considered problems here.

D. If information is missing or unclear, leave the respective field empty.

RETURN ONLY THE JSON – no markdown fences, no explanatory text.
""")





# create chunkAchievements
promptAchievements = ("""
You are an expert information-extractor.
Your task: from the provided diary entry create structured “Achievement” objects that match the supplied list of descriptions.

Definitions – read carefully
• Achievement = a completed goal, success or positive outcome already obtained (personal, professional, social, etc.).
• It must be in the past or present. Do NOT include plans, wishes, intentions or anything that has not happened yet.

Input variables
diary_entry      → a single free-text diary chunk.
descriptions     → lines of descriptions. Each line describes a piece of information I want transformed into an Achievement.

Extraction rules

For every element in descriptions produce exactly ONE Achievement object.
If two (or more) description strings obviously refer to the SAME real-world achievement, MERGE them into one object (therefore the final number of objects can be smaller than lines of descriptions).
If the diary gives no clear evidence for a description, still output an Achievement object for it but leave all fields empty (empty string or empty list).
Only extract what is explicitly present or can be deduced with high certainty from the diary; never hallucinate.
Field-level instructions (schema given below)
• content  – 1–2 short sentences that answer: “What was achieved?”
• title    – one or two words, capitalised. No more.
• people   – list of names (or “Mom”, “Boss”, etc.) that are strongly connected to the achievement. Empty list if none.
• emotions – list of emotion words that are explicitly stated or unmistakably implied in context with that achievement. Empty list if none.
""")








# create chunkFutures
promptGoals = ("""
You are given:

1. diary_entry  – one diary chunk written by the user.
2. descriptions – each line tells which future-oriented
   achievement the user wants us to capture.

Task
Create a JSON object that validates against the schema
defined.

Rules
• Produce JSON only – no prose, no markdown, no explanations.  
• For every element/line in descriptions create exactly one
  `FutureIntention` item, unless two elements clearly describe
  the same achievement, in which case merge them into a single item.  
• Strictly follow the field definitions:

  - content: A concise sentence that states what will / should happen.  
  - title  : 1–2 words, Title Case, no punctuation.  
  - people : List of names that are strongly connected to the intention.
             Omit nicknames in quotes, keep first names or full names
             exactly as written in the diary.  
             If the diary gives no clear person, return an empty list.

• Leave a field empty ("" for strings, [] for lists) if the diary
  chunk gives insufficient evidence.

• DO NOT invent information that is not explicitly or implicitly
  present in the diary entry.
""")







# EVALUATION

promptEvalPrompt = ("""
SYSTEM:
You are a strict quality evaluator for a Retrieval-Augmented-Generation (RAG) system.
The passage you will judge may be in English **or German**.
Score it according to the rubric below and return ONLY the JSON specified in OutputEval.

CRITERIA:
1. Relevance (0–2)
   • 0 = no relation to the question  
   • 1 = some topical terms, little substance  
   • 2 = concrete facts/arguments that help answer  
2. Overall Utility (0–2)
   • 0 = no added value for answering  
   • 1 = useful extra info but not essential  
   • 2 = key information without which the answer would be incomplete



""")





# GENERATION

finalGenerationPrompt = """
task: answer the users message using the provided context: 

"""
