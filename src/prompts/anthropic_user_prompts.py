COMMENT_PROMPT_ANTHROPIC = """
<discussion_context>
<title>{discussion_title}</title>
<summary>{discussion_summary}</summary>
<discussion_history>{discussion_history}</discussion_history>
</discussion_context>
<task>
Write a social media comment on this discussion topic.
IMPORTANT: Write your comment in the same language as the discussion title. Ensure correct grammar and syntax for that language.
</task>
<formatting_constraints>
- Write as natural social media text only - NO markdown headers (no #, ##, etc.)
- Do NOT structure your comment with headers or sections
- Write in continuous prose as you would in a typical social media comment
- Avoid any markdown formatting like bold (**text**), headers, or bullet points unless absolutely natural for the platform
- Your output should be plain text that could be posted directly to a social media platform
</formatting_constraints>
<content_constraints>
- Make your comment DISTINCT from existing comments in the discussion history
- Do NOT paraphrase, echo, or closely mirror the language used in previous comments
- Do NOT mention other users by name
- Bring a fresh perspective, angle, or phrasing even if your viewpoint is similar to others
- Avoid repeating the same examples, phrases, or arguments already mentioned
- Contribute something new to the conversation - a different emphasis, a unique concern, or an original way of expressing your position
</content_constraints>"""

REPLY_PROMPT_ANTHROPIC = """
<discussion_context>
<title>{discussion_title}</title>
<summary>{discussion_summary}</summary>
<discussion_history>{discussion_history}</discussion_history>
</discussion_context>
<candidate_posts>
{candidate_posts}
</candidate_posts>
<task>
Choose ONE of the candidate posts above to reply to. Begin your response with the POST_ID you are replying to on its own line, e.g. [POST_ID=abc-123], then write your reply.

Your reply should:
1. Directly address the points raised in the chosen post
2. Maintain the overall discussion context
3. Reflect your perspective and contribute meaningfully
IMPORTANT: Write your reply in the same language as the discussion title. Ensure correct grammar and syntax for that language.
</task>
<formatting_constraints>
- Write as natural social media text only - NO markdown headers (no #, ##, etc.)
- Do NOT structure your reply with headers or sections
- Write in continuous prose as you would in a typical social media reply
- Avoid any markdown formatting like bold (**text**), headers, or bullet points unless absolutely natural for the platform
- Your output should be plain text that could be posted directly to a social media platform
</formatting_constraints>
<content_constraints>
- Make your reply DISTINCT from existing comments in the discussion history
- Do NOT paraphrase, echo, or closely mirror the language used in previous comments
- While responding to the target post, use your own voice and phrasing
- Avoid repeating the same examples, phrases, or arguments already mentioned by others
- Bring your unique perspective even when agreeing or disagreeing with the target post
- Contribute original thoughts rather than restating what's already been said
</content_constraints>"""


ACTION_PROMPT_ANTHROPIC = """
<eligible_posts>
{eligible_posts}
</eligible_posts>
<action_history>
{action_history}
</action_history>
<task>
You are reviewing posts in an online discussion. Select up to {m_candidates} posts from the eligible posts above that you feel most strongly about (positively or negatively).

For each selected post, decide whether to upvote or downvote it.

Respond with ONLY a JSON array. No markdown, no explanation, no code blocks. Example:
[
  {{"post_id": "some-id", "vote_type": "upvote"}},
  {{"post_id": "another-id", "vote_type": "downvote"}}
]
</task>"""
