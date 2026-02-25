# ══════════════════════════════════════════════════════════════
# AGENTIC AI TAB (SIMPLIFIED)
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🤖 Agentic AI — Autonomous Project Scanning")
    st.caption("The agent automatically runs all 5 models across your full dataset, chains findings together, and surfaces prioritized actions — no manual input needed.")

    with st.spinner("🧠 Agent scanning dataset across all objectives..."):
        findings = run_agent_scan(df, list(models.keys()))
        chain    = build_chain(findings)
        score    = compute_health(findings)

    # ── Health Score ────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        color = "#06d6a0" if score >= 75 else ("#ffd166" if score >= 50 else "#ff4d6d")
        label = "🟢 Healthy" if score >= 75 else ("🟡 Needs Attention" if score >= 50 else "🔴 At Risk")
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:3.5rem;font-weight:900;color:{color};'>{score}</div>
            <div style='font-size:1rem;color:#aaa;margin-top:-8px;'>/ 100</div>
            <div style='font-size:1.1rem;margin-top:6px;'>{label}</div>
            <div style='font-size:0.8rem;color:#888;'>Project Health Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        bar_color = "#06d6a0" if score >= 75 else ("#ffd166" if score >= 50 else "#ff4d6d")
        st.markdown(f"""
        <div style='margin-top:2rem;'>
            <div class='health-bar-container'>
                <div style='background:{bar_color};width:{score}%;height:14px;border-radius:8px;transition:width 1s ease;'></div>
            </div>
            <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-top:4px;'>
                <span>0 — Critical</span><span>50 — Attention</span><span>100 — Healthy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        criticals_n = sum(1 for f in findings if f['severity']=='critical')
        warnings_n  = sum(1 for f in findings if f['severity']=='warning')
        st.markdown(f"""
        <div style='text-align:center;margin-top:0.5rem;'>
            <div style='font-size:2rem;font-weight:800;color:#ff4d6d;'>{criticals_n}</div>
            <div style='font-size:0.8rem;color:#aaa;'>Critical Issues</div>
            <div style='font-size:2rem;font-weight:800;color:#ffd166;margin-top:8px;'>{warnings_n}</div>
            <div style='font-size:0.8rem;color:#aaa;'>Warnings</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Findings ────────────────────────────────────────────
    st.subheader("🔍 Autonomous Findings")
    severity_order = {'critical': 0, 'warning': 1, 'info': 2, 'success': 3}

    for f in sorted(findings, key=lambda x: severity_order.get(x['severity'], 99)):
        action_html = f"<div style='margin-top:6px;font-style:italic;opacity:0.75;'>→ {f['action']}</div>" if f['action'] else ""
        st.markdown(f"""
        <div class='agent-card {f["severity"]}'>
            <div class='agent-title'>{f["icon"]} [{f["objective"]}] {f["title"]}</div>
            <div class='agent-detail'>{f["detail"]}{action_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Decision Chain ──────────────────────────────────────
    st.subheader("⛓️ Chained Decision Reasoning")
    st.caption("The agent links findings across objectives to produce connected, prioritized recommendations.")

    for step in chain:
        icon   = "✅" if step['status'] == 'done' else ("⚠️" if step['status'] == 'alert' else "💡")
        color  = "#4cc9f0" if step['status'] == 'done' else ("#ffd166" if step['status'] == 'alert' else "#06d6a0")
        sev    = "info" if step['status'] == 'done' else ("warning" if step['status'] == 'alert' else "success")

        st.markdown(f"""
        <div class='agent-card {sev}' style='display:flex;gap:1rem;align-items:flex-start;'>
            <div style='background:{color};color:#000;border-radius:50%;width:30px;height:30px;
                        display:flex;align-items:center;justify-content:center;
                        font-weight:800;font-size:0.8rem;flex-shrink:0;margin-top:2px;'>
                {step["step"]}
            </div>
            <div>
                <div class='agent-title'>{icon} {step["label"]}</div>
                <div class='agent-detail'>{step["detail"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
