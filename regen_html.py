import json
import solution

with open('emotion_timeline_output.json', 'r') as f:
    stats = json.load(f)

frame_series = stats['frame_series']
micro_expressions = stats['micro_expressions']
transitions = stats['transitions']

# Re-generate the HTML instantly
solution.generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, solution.REPORT_HTML_OUT)
print('Regenerated HTML.')
