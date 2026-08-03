# Model Card: Music Matcher — RAG-Integrated Music Recommender

## Limitations and biases
- What are the limitations or biases in your system?
My system is limited to scoring songs strictly based on the feature weights I chose. This means
that songs which are good fits to the prompt but fall into different perceived genres score low. I keep
the scoring logic this way to maintain a simple understanding of the app's scoring formula and easily
tweak it.
The local "taste codebook" document is retrieved by the AI before creating a preferences dictionary
for the user. Unfortunately, the unvariable codebook is bound to include cultural biases on how to
interpret natural language (e.g., a rainy day may imply a mellow mood in one culture while implying
an upbeat mood in another).

## AI Misuse
- Could your AI be misused, and how would you prevent that?
I believe my AI cannot be easily misused: the grounding guardrail already prevents hallucinated/fake song recommendations, the Streamlit BYO-key and offline-default design avoids storing or exposing credentials, and the deterministic core bounds what the AI can output.

## AI Reliability Surprises
- What surprised you while testing your AI's reliability?
I was surprised by the sensitivity my AI has to small changes in user prompts. However, the logic behind
LLMs makes the cause of the sensitivity clear. LLMs are not deterministic, meaning even the same exact prompt
outputs differently every run. By adding a change in prompts, the AI's predictive engine now also has
a changed start to build its nondeterministic logic on. The AI's sensitivity to prompt changes reminds
me of a butterfly effect that is only mitigated through increasing details in your prompting.

## AI Collaboration
- Describe your collaboration with AI during this project.
I utilized Claude Code during this project to build the app logic and tests corresponding to the detailed system
design prompts I gave it. I relied on it very little for creativity in the app's design and when obstacles
were struck, I made instructions for Claude to carefully explain the issue and not enact changes until
I decide the best course of action. This method of collaboration allowed me to stay true to my goal system
design while speeding the creation of the internal code logic up.

## Helpful/Flawed AI Recommendations
- Identify one instance when the AI gave a helpful suggestion and one instance where its suggestion was flawed or 
  incorrect.
The AI gave a very helpful suggestion in splitting the "propose songs" step into "search freely" then "format
with a schema-guaranteed structured-output call." This logic allowed retrieved songs to still be verified while
not excluding real songs that have "a lack in their accompanying musical information" and causing a fallback
to offline mode.
When Anthropic API mode was first failing in producing AI-generated, online outputs, I consulted Claude Code in 
finding the logic failure. Claude gave me a multitude of potentially untested app logic steps that may have to be
re-verified. This re-verification process lasted minutes with no real culprits found until I realized that I
had no free trial in API credits from Anthropic. My Claude Code had no way of knowing that was the case, but the 
confidence it gave in its answers failed to point out other possibilities.