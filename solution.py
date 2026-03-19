import cv2
import json
import numpy as np
from pathlib import Path
import face_recognition
from deepface import DeepFace

VIDEO_PATH = Path("Dataset_Assignment/Video_1/Class_8_cctv_video_1.mov")
PROFILE_IMAGE = Path("Dataset_Assignment/Profiles_1/Harshita.png")

REPORT_HTML_OUT = Path("emotion_timeline.html")
OUTPUT_JSON = Path("emotion_timeline_output.json")

ANALYSIS_FPS = 8
MICRO_MAX_SEC = 0.5
MICRO_MIN_PROB = 0.40
NEUTRAL_MIN = 0.50

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# Load Harshita profile encoding
profile_img = face_recognition.load_image_file(PROFILE_IMAGE)
profile_enc = face_recognition.face_encodings(profile_img)[0]


def detect_primary_face(frame):

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)

    encodings = face_recognition.face_encodings(rgb,locations)

    for (top,right,bottom,left),enc in zip(locations,encodings):

        match = face_recognition.compare_faces([profile_enc],enc,tolerance=0.5)

        if match[0]:

            crop = frame[top:bottom,left:right]

            return crop,(left,top,right-left,bottom-top)

    return None,None


def analyse_emotion(face):

    try:

        result = DeepFace.analyze(
            face,
            actions=["emotion"],
            enforce_detection=False
        )

        emo = result[0]["emotion"] if isinstance(result,list) else result["emotion"]

        total=sum(emo.values()) or 1

        probs={}

        for e in EMOTIONS:
            probs[e]=round(emo.get(e,0)/total*100,2)

        return probs

    except:

        return {e:(100 if e=="neutral" else 0) for e in EMOTIONS}


def detect_micro_expressions(frame_series):

    micro=[]

    max_frames=int(MICRO_MAX_SEC*ANALYSIS_FPS)

    i=1
    eid=1

    while i<len(frame_series)-1:

        cur=frame_series[i]
        prev=frame_series[i-1]

        dom=max(cur["emotions"],key=cur["emotions"].get)

        prob=cur["emotions"][dom]

        if dom!="neutral" and prob>=MICRO_MIN_PROB*100:

            if prev["emotions"]["neutral"]>=NEUTRAL_MIN*100:

                start=i
                j=i

                while j<len(frame_series) and frame_series[j]["emotions"][dom]>=MICRO_MIN_PROB*100:
                    j+=1

                duration=j-start

                if duration<=max_frames and frame_series[j]["emotions"]["neutral"]>=NEUTRAL_MIN*100:

                    micro.append({
                        "id":eid,
                        "timestamp_sec":frame_series[start]["t"],
                        "duration_sec":duration/ANALYSIS_FPS,
                        "emotion":dom,
                        "peak_probability":prob,
                        "followed_by":"neutral",
                        "is_suppressed":True
                    })

                    eid+=1

                i=j
                continue

        i+=1

    return micro


def detect_transitions(frame_series):

    transitions=[]

    for i in range(1,len(frame_series)):

        prev=frame_series[i-1]
        cur=frame_series[i]

        prev_dom=max(prev["emotions"],key=prev["emotions"].get)
        cur_dom=max(cur["emotions"],key=cur["emotions"].get)

        if prev_dom!=cur_dom:

            transitions.append({
                "from_emotion":prev_dom,
                "to_emotion":cur_dom,
                "timestamp_sec":cur["t"],
                "transition_duration_sec":0.5
            })

    return transitions


def compute_suppression_score(frame_series,micro):

    events=0

    for fs in frame_series:
        if any(v>35 for k,v in fs["emotions"].items() if k!="neutral"):
            events+=1

    if events==0:
        return 0

    return int((len(micro)/events)*100)


def compute_emotional_range(frame_series):

    seen=set()
    probs=[]

    for fs in frame_series:
        for e,p in fs["emotions"].items():

            if p>30:
                seen.add(e)

            probs.append(p)

    distinct=len(seen)

    std=np.std(probs)

    score=min(100,(distinct/7)*100 + std*2)

    return int(score)


def generate_emotion_timeline_html(frame_series,stats):

    times=[f["t"] for f in frame_series]

    datasets=""

    for emo in EMOTIONS:

        values=[f["emotions"][emo] for f in frame_series]

        datasets+=f"""
        {{
        label: "{emo}",
        data: {values},
        fill: true
        }},
        """

    html=f"""
<html>
<head>
<title>Emotion Timeline</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>

<h2>Emotion Timeline (Harshita)</h2>

<canvas id="chart"></canvas>

<script>

const data={{

labels:{times},

datasets:[{datasets}]

}}

new Chart(
document.getElementById('chart'),
{{type:'line',data:data}}
)

</script>

<h3>Statistics</h3>

<pre>{json.dumps(stats,indent=2)}</pre>

</body>
</html>
"""

    with open(REPORT_HTML_OUT,"w") as f:
        f.write(html)


if __name__=="__main__":

    cap=cv2.VideoCapture(str(VIDEO_PATH))

    fps=cap.get(cv2.CAP_PROP_FPS) or 25

    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration=total/fps

    sample=max(1,int(fps/ANALYSIS_FPS))

    frame_series=[]

    frame_idx=0

    while True:

        ret,frame=cap.read()

        if not ret:
            break

        if frame_idx%sample==0:

            ts=frame_idx/fps

            face,_=detect_primary_face(frame)

            if face is not None:
                emotions=analyse_emotion(face)
            else:
                emotions={e:(100 if e=="neutral" else 0) for e in EMOTIONS}

            frame_series.append({
                "t":round(ts,3),
                "emotions":emotions
            })

        frame_idx+=1

    cap.release()

    micro=detect_micro_expressions(frame_series)

    transitions=detect_transitions(frame_series)

    suppression=compute_suppression_score(frame_series,micro)

    emo_range=compute_emotional_range(frame_series)

    counts={e:0 for e in EMOTIONS}

    for fs in frame_series:
        dom=max(fs["emotions"],key=fs["emotions"].get)
        counts[dom]+=1

    n=len(frame_series)

    pct={e:round(c/n*100,1) for e,c in counts.items()}

    dominant=max(pct,key=pct.get)

    stats={
        "source":"p6_emotion_timeline",
        "video":str(VIDEO_PATH),
        "duration_sec":round(duration,2),
        "fps_analyzed":ANALYSIS_FPS,
        "dominant_emotion":dominant,
        "emotion_time_pct":pct,
        "suppression_score":suppression,
        "emotional_range_score":emo_range,
        "micro_expressions":micro,
        "transitions":transitions,
        "frame_series":frame_series
    }

    with open(OUTPUT_JSON,"w") as f:
        json.dump(stats,f,indent=2)

    generate_emotion_timeline_html(frame_series,stats)

    print("Report generated")