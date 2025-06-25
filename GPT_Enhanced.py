# GenAI SAP Support Ticket Dashboard (Corrected Analyzer Stage Visibility + Ticket Distribution)

import streamlit as st
import pandas as pd
import time
import uuid
import os
import random
from datetime import datetime
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

# === CONFIG ===
st.set_page_config(layout="wide")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st_autorefresh(interval=10000, limit=None, key="autorefresh")

# === SESSION STATE ===
if "ticket_queue" not in st.session_state:
    st.session_state.ticket_queue = []
if "completed_tickets" not in st.session_state:
    st.session_state.completed_tickets = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# === LOAD TICKETS ===
@st.cache_data
def load_initial_tickets():
    df = pd.read_csv("sap_issues_full_100.csv")
    initial = df.head(10)
    tickets = []
    total = len(initial)
    min_to_analyze = total // 3
    for i, (_, row) in enumerate(initial.iterrows()):
        force_analysis = i < min_to_analyze
        is_known = False if force_analysis else random.choices([True, False], weights=[6, 4])[0]
        tickets.append({
            "id": row["Ticket ID"],
            "text": row["Issue Summary"],
            "is_known": is_known,
            "status": "New",
            "assigned_agent": "Triaging Agent",
            "agent_history": [],
            "stage": "Queued for triage",
            "start_time": datetime.now(),
            "end_time": None,
            "whys": [],
            "log_findings": "",
            "root_cause": "",
            "8d_report_path": None,
            "was_analyzed": False,
        })
    return tickets

if not st.session_state.ticket_queue:
    st.session_state.ticket_queue = load_initial_tickets()

# === GENAI 5 WHY ===

    # Load enhanced ticket context CSV
@st.cache_data
def load_ticket_context():
    return pd.read_csv("sap_ticket_enhanced_context.csv", dtype=str).set_index("Ticket ID")

ticket_context_df = load_ticket_context()

# Enhanced 5 Why Analysis with Context
def get_gpt_root_cause(ticket_id, issue_text):
    try:
        context = ticket_context_df.loc[ticket_id] if ticket_id in ticket_context_df.index else None
        context_prompt = ""

        if context is not None:
            context_prompt = f"""
Additional Ticket Context:
- SAP Module / Transaction Code: {context.get('Transaction Code', 'N/A')}
- Error Message / Log: {context.get('Error Message', 'N/A')}
- Affected Business Process: {context.get('Business Process', 'N/A')}
- Previous Fix Attempts: {context.get('Previous Fixes', 'None')}
- Metadata: {context.get('Metadata', 'N/A')}
"""

        prompt = f"""You are an SAP support incident analyst. Perform a 5 Whys analysis based on the issue and context below.

Issue Summary: {issue_text}
{context_prompt}

Respond with 5 Whys in bullet points followed by a clear final root cause."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an SAP incident expert skilled in root cause analysis and KB resolution."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()

        import re
        lines = reply.splitlines()
        whys = []
        for line in lines:
            line = line.strip()
            if re.match(r"^[-*]\s*", line) and "why" in line.lower():
                whys.append(line.lstrip("-* ").strip())
            elif re.match(r"(?i)^why\s*\d*[:\.-]?", line):
                whys.append(re.sub(r"(?i)^why\s*\d*[:\.-]?\s*", "", line).strip())

        whys = whys[:5]
        while len(whys) < 5:
            whys.append("(Placeholder to reach 5 Whys)")

        root_cause = whys[-1] if whys and "Placeholder" not in whys[-1] else "Root cause unknown"
        return whys, root_cause
    except Exception as e:
        return [f"Error: {str(e)}"], "Root cause unknown"



# === ADVANCE TICKETS ===
def change_agent(ticket, new_agent):
    if ticket["assigned_agent"] != new_agent:
        ticket.setdefault("agent_history", []).append(ticket["assigned_agent"])
        ticket["assigned_agent"] = new_agent

def advance_tickets():
    for ticket in st.session_state.ticket_queue:
        if ticket["status"] == "Closed":
            continue
        if ticket["assigned_agent"] == "Triaging Agent":
            if ticket["stage"] == "Queued for triage":
                ticket["stage"] = "Classifying issue"
            else:
                if ticket["is_known"]:
                    ticket["status"] = "Closed"
                    ticket["stage"] = "Resolved via KB"
                    ticket["end_time"] = datetime.now()
                    st.session_state.completed_tickets.append(ticket)
                else:
                    change_agent(ticket, "Analyzer Agent")
                    ticket["stage"] = "Parsing logs"

        elif ticket["assigned_agent"] == "Analyzer Agent":
            if ticket["stage"] == "Parsing logs":
                ticket["stage"] = "Running 5 Whys"
                ticket["log_findings"] = "SAP logs reveal error in background job due to missing config."

            elif ticket["stage"] == "Running 5 Whys":
                whys, root_cause = get_gpt_root_cause(ticket["id"], ticket["text"])
                ticket["whys"] = whys
                ticket["root_cause"] = root_cause
                ticket["was_analyzed"] = True
                ticket["stage"] = "Suggesting fix"

            elif ticket["stage"] == "Suggesting fix":
                change_agent(ticket, "8D Agent")
                ticket["stage"] = "Creating 8D report"

        elif ticket["assigned_agent"] == "8D Agent":
            if ticket["stage"] == "Creating 8D report":
                output_folder = "output_reports"
                os.makedirs(output_folder, exist_ok=True)
                file_path = os.path.join(output_folder, f"8D_{ticket['id']}.xlsx")
                df = pd.DataFrame({
                    "Discipline": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
                    "Description": [
                        "SAP Module", "Transaction Code", "Incident Summary", "Root Cause",
                        "Business Impact", "Corrective Action", "Validation Method", "Closure Remarks"
                    ],
                    "Details": [
                        "Auto-detected", "N/A", ticket["text"], ticket["root_cause"] if "Placeholder" not in ticket["root_cause"] else "Root cause pending analysis",
                        "Business interruption", "Config fix applied", "Validated in QA", "Case closed"
                    ]
                })
                df.to_excel(file_path, index=False)
                ticket["8d_report_path"] = file_path
                ticket["stage"] = "Updating KB"
            else:
                ticket["status"] = "Closed"
                ticket["stage"] = "Completed"
                ticket["end_time"] = datetime.now()
                st.session_state.completed_tickets.append(ticket)

# === RENDER UI ===
def render_tabs():
    df = pd.DataFrame(st.session_state.ticket_queue + st.session_state.completed_tickets)
    df.sort_values("start_time", ascending=False, inplace=True)
    tabs = st.tabs(["Ticket Summary", "Ticket Movement View", "Analysis Agent Tickets", "KPI Dashboard"])

    with tabs[0]:
        st.subheader("Recent Tickets")
        page_size = 10
        total_pages = (len(df) - 1) // page_size + 1
        start = st.session_state.current_page * page_size
        end = start + page_size
        paginated = df.iloc[start:end]
        if not paginated.empty:
            st.dataframe(paginated[["id", "text", "assigned_agent", "stage", "status", "8d_report_path"]]
                         .rename(columns={"id": "Ticket ID", "text": "Summary", "assigned_agent": "Agent",
                                          "stage": "Activity", "status": "Status", "8d_report_path": "8D Report"}),
                         use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("\u2b05\ufe0f Previous") and st.session_state.current_page > 0:
            st.session_state.current_page -= 1
        if col2.button("Next \u2794") and st.session_state.current_page < total_pages - 1:
            st.session_state.current_page += 1

    with tabs[1]:
        st.subheader("Ticket Movement View")
        cols = st.columns(3)
        agent_map = {"Triage Agent": "Triaging Agent", "Analysis Agent": "Analyzer Agent", "8D Agent": "8D Agent"}
        df_active = df.copy()
        for i, (title, internal) in enumerate(agent_map.items()):
            with cols[i]:
                st.markdown(f"### {title}")
                sub_df = df_active[df_active["assigned_agent"] == internal][["id", "text", "stage"]]
                sub_df.rename(columns={"id": "Ticket Number", "text": "Summary",
                                       "stage": "Activity in Progress"}, inplace=True)
                if not sub_df.empty:
                    st.dataframe(sub_df.style.apply(
                        lambda row: ['background-color: lightgreen' if row.get('Activity in Progress') == 'Closed' else '' for _ in row], axis=1),
                        use_container_width=True)
                else:
                    st.info("No active tickets.")

    with tabs[2]:
        st.subheader("Analysis Agent Tickets")
        df_analysis = df[df.apply(lambda row: row.get("assigned_agent") == "Analyzer Agent" or "Analyzer Agent" in row.get("agent_history", []), axis=1)]
        if not df_analysis.empty:
            selected_id = st.selectbox("Select Analysis Ticket", df_analysis["id"])
            ticket = df_analysis[df_analysis["id"] == selected_id].iloc[0]
            st.markdown(f"### Details for {ticket['id']}")
            st.markdown(f"**Summary:** {ticket['text']}")
            st.markdown(f"**Agent:** {ticket['assigned_agent']}")
            st.markdown(f"**Stage:** {ticket['stage']}")
            st.markdown(f"**Status:** {ticket['status']}")
            if ticket["log_findings"]:
                st.markdown(f"**Log Analyzer Findings:** {ticket['log_findings']}")
            if ticket["whys"]:
                st.markdown("**5 Whys Analysis:**")
                st.markdown("<ol>" + "".join([f"<li>{why}</li>" for why in ticket["whys"]]) + "</ol>", unsafe_allow_html=True)
                if not any("Placeholder" in why for why in ticket["whys"]):
                    st.markdown(f"**Identified Root Cause:** {ticket['whys'][-1]}")
            else:
                st.warning("Root cause still being refined. Please review analysis.")
            if ticket["root_cause"] and "Placeholder" not in ticket["root_cause"]:
                st.markdown(f"**Final Root Cause:** {ticket['root_cause']}")
            if ticket["8d_report_path"]:
                st.markdown(f"[Download 8D Report]({ticket['8d_report_path']})")

    with tabs[3]:
        st.subheader("KPI Dashboard")
        df_closed = df[df["status"] == "Closed"]
        df_8d = df[df["8d_report_path"].notnull()]
        df_kb = df_closed[df_closed["stage"] == "Resolved via KB"]
        df_analysis = df[df["was_analyzed"] == True]
        df_in_progress = df[df["status"] != "Closed"]

        def avg_resolution_time(df):
            durations = [(row["end_time"] - row["start_time"]).total_seconds()/60 for _, row in df.iterrows() if row["end_time"]]
            return round(sum(durations)/len(durations), 2) if durations else 0

        kpi_data = [
            ["Total Tickets Processed", len(df), "count", "tickets"],
            ["Average Resolution Time", avg_resolution_time(df_closed), "average time", "minutes"],
            ["Tickets Closed via KB", len(df_kb), "count", "tickets"],
            ["Tickets Escalated to 8D Agent", len(df[df['assigned_agent'] == '8D Agent']), "count", "tickets"],
            ["8D Reports Generated", len(df_8d), "count", "reports"],
            ["Triage Effectiveness", round((len(df_kb)/len(df))*100, 2) if len(df) else 0, "percentage", "%"],
            ["Analysis Depth Score", round(df_analysis['whys'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean(), 2) if not df_analysis.empty else 0, "average depth", "Whys"],
            ["8D Completion Rate", round((len(df_8d)/len(df[df['assigned_agent'] == '8D Agent']))*100, 2) if len(df[df['assigned_agent'] == '8D Agent']) else 0, "percentage", "%"],
            ["Tickets in Queue", len(df_in_progress), "count", "tickets"],
            ["Tickets in Progress", len(df[(df['stage'] != 'Completed') & (df['status'] != 'Closed')]), "count", "tickets"],
            ["Stage Bottleneck", df['stage'].value_counts().idxmax() if not df.empty else "N/A", "stage", "n/a"]
        ]
        st.dataframe(pd.DataFrame(kpi_data, columns=["KPI", "Value", "Type", "Unit"]), use_container_width=True)

render_tabs()
advance_tickets()
