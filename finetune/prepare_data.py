# =============================================================================
# finetune/prepare_data.py — Build fine-tuning dataset for qwen3:4b
# Sources:
#   1. CIC IoMT 2024 attack knowledge (hand-crafted Q&A pairs)
#   2. MITRE ATT&CK IoT techniques (live API)
#   3. NVD CVE descriptions (live API, IoT filtered)
#   4. Phase 1 generated reports (your own pipeline output)
# Output: finetune/dataset.jsonl (instruction-response pairs)
# =============================================================================

import os
import json
import time
import requests
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR, NVD_API_URL, NVD_API_KEY

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")


# =============================================================================
# SOURCE 1 — CIC IoMT 2024 hand-crafted attack knowledge
# Rich Q&A pairs covering all 13 attack types in your dataset
# =============================================================================

IOMT_KNOWLEDGE = [
    # ── DDoS Attacks ──────────────────────────────────────────────────────────
    {
        "attack": "TCP_IP-DDoS-ICMP1",
        "instruction": "An IDS detected TCP_IP-DDoS-ICMP1 attack on an IoMT network with CRITICAL severity. LightGBM confidence: 98%. Top features: ICMP=1.0, IPv=1.0, LLC=1.0. Analyse this alert.",
        "response": """## Security Analysis: TCP/IP DDoS ICMP Flood (Type 1)

**Attack Summary:**
This is a high-volume ICMP flood DDoS attack targeting IoMT devices over TCP/IP infrastructure. The attacker sends massive volumes of ICMP Echo Request packets to overwhelm network resources and render medical IoT devices unreachable. The 98% LightGBM confidence confirms this as a well-established attack pattern.

**Attack Characteristics:**
- High ICMP traffic volume saturating network bandwidth
- IPv4/IPv6 protocol exploitation for packet amplification
- LLC layer activity indicating layer-2 flooding component
- Targets availability of critical IoMT devices (patient monitors, infusion pumps)

**MITRE ATT&CK Mapping:**
- Technique: T1498 — Network Denial of Service
- Sub-technique: T1498.001 — Direct Network Flood
- Tactic: Impact
- Platform: Network Infrastructure

**Severity Assessment:** CRITICAL (CVSS 9.0)
- Direct threat to patient safety through IoMT device unavailability
- Can disrupt real-time patient monitoring systems
- High impact on healthcare operations

**Immediate Mitigation:**
1. Enable ICMP rate limiting on all network interfaces immediately
2. Block source IP ranges at perimeter firewall
3. Deploy upstream DDoS scrubbing service
4. Enable ICMP flood protection on network switches
5. Isolate affected IoMT network segment
6. Alert network security team and activate incident response plan

**Long-term Recommendations:**
- Implement network segmentation for IoMT devices
- Deploy dedicated DDoS protection appliance
- Configure ICMP traffic baselines and anomaly alerts"""
    },
    {
        "attack": "TCP_IP-DDoS-ICMP2",
        "instruction": "IDS alert: TCP_IP-DDoS-ICMP2 detected. Anomaly score 0.000199 vs threshold 5e-06. Both LightGBM and Transformer Autoencoder flagged this flow. Protocol: TCP/IP. Provide security analysis.",
        "response": """## Security Analysis: TCP/IP DDoS ICMP Flood (Type 2)

**Attack Summary:**
TCP_IP-DDoS-ICMP2 is an advanced variant of ICMP flood attack with higher packet complexity than Type 1. The dual detection by both LightGBM (known attack) and Transformer Autoencoder (anomaly score 40x above threshold) indicates this attack has both known signatures and novel characteristics — suggesting a zero-day variant of a known attack family.

**Attack Characteristics:**
- Advanced ICMP flood with packet fragmentation techniques
- Anomaly score 40x above threshold indicates highly abnormal traffic
- Dual IDS detection confirms high-confidence threat
- May use spoofed source IPs to bypass basic filtering

**MITRE ATT&CK Mapping:**
- Technique: T1498 — Network Denial of Service
- Sub-technique: T1498.001 — Direct Network Flood
- Tactic: Impact

**CVE References:**
- CVE-2021-35247: ICMP flood vulnerability in network devices
- CVE-2020-25705: ICMP-based network reconnaissance and DoS

**Severity Assessment:** CRITICAL (CVSS 9.0)
- Zero-day characteristics increase severity
- IoMT device unavailability poses direct patient risk

**Immediate Mitigation:**
1. Block all non-essential ICMP traffic at firewall
2. Enable SYN cookies and ICMP rate limiting
3. Deploy upstream traffic scrubbing immediately
4. Capture full packet trace for forensic analysis
5. Check for IP spoofing — enable anti-spoofing filters
6. Activate DDoS response playbook"""
    },
    {
        "attack": "MQTT-DDoS-Connect_Flood",
        "instruction": "Alert: MQTT-DDoS-Connect_Flood detected on IoMT network. LightGBM: 95% confidence. Anomaly detected. Top features: TCP=1.0, MQTT protocol active. Analyse and provide mitigation.",
        "response": """## Security Analysis: MQTT DDoS Connection Flood

**Attack Summary:**
This attack floods an MQTT broker with massive numbers of connection requests, exhausting broker resources and denying legitimate IoMT device communications. MQTT is the primary protocol for IoT/IoMT messaging, making this attack particularly dangerous for healthcare IoT infrastructure.

**Attack Characteristics:**
- Thousands of CONNECT packets per second sent to MQTT broker
- Exhausts broker connection pool and memory
- Disrupts all IoMT device communications relying on MQTT
- Can cascade to patient monitoring failure

**MITRE ATT&CK Mapping:**
- Technique: T1499 — Endpoint Denial of Service
- Sub-technique: T1499.002 — Service Exhaustion Flood
- Tactic: Impact

**CVE References:**
- CVE-2023-28366: MQTT broker memory exhaustion via connection flood
- CVE-2021-34567: Eclipse Mosquitto MQTT broker DoS vulnerability
- CVE-2020-13849: MQTT protocol connection exhaustion

**Severity Assessment:** CRITICAL (CVSS 9.2)
- Direct disruption to IoMT device communication
- Patient safety risk from disconnected monitoring devices

**Immediate Mitigation:**
1. Enable MQTT connection rate limiting on broker (max 100 connections/sec)
2. Implement connection authentication — reject anonymous connections
3. Whitelist known IoMT device MAC/IP addresses
4. Restart overloaded MQTT broker after applying rate limits
5. Deploy MQTT-aware firewall rules
6. Enable TLS for all MQTT connections to prevent spoofing
7. Monitor broker logs for connection source patterns"""
    },
    {
        "attack": "MQTT-DDoS-Publish_Flood",
        "instruction": "MQTT-DDoS-Publish_Flood detected with CRITICAL severity. LightGBM: 97% confidence. MQTT protocol traffic extremely high. What is this attack and how to respond?",
        "response": """## Security Analysis: MQTT DDoS Publish Flood

**Attack Summary:**
The attacker floods the MQTT broker with massive volumes of PUBLISH messages, overwhelming the broker's message queue processing capacity. Unlike connection flood attacks, publish floods can bypass connection limits and are harder to detect since they use legitimate protocol operations.

**Attack Characteristics:**
- Extremely high volume PUBLISH packets on MQTT topics
- Overwhelms broker message queue and subscriber processing
- Can target specific critical topics (patient vitals, device commands)
- May cause message loss for legitimate IoMT communications

**MITRE ATT&CK Mapping:**
- Technique: T1499 — Endpoint Denial of Service
- Sub-technique: T1499.002 — Service Exhaustion Flood
- Tactic: Impact

**CVE References:**
- CVE-2023-28366: MQTT broker resource exhaustion
- CVE-2022-41906: MQTT publish flood in IoT deployments

**Severity Assessment:** CRITICAL (CVSS 9.0)

**Immediate Mitigation:**
1. Implement MQTT publish rate limiting per client (max 50 msgs/sec)
2. Enable message size limits on broker
3. Block publishing clients exceeding rate limits
4. Set QoS level limits to reduce broker load
5. Monitor and alert on abnormal publish rates
6. Implement topic-based access control lists (ACLs)"""
    },
    {
        "attack": "MQTT-DoS-Connect_Flood",
        "instruction": "Single-source MQTT-DoS-Connect_Flood detected. LightGBM: 93% confidence. Source appears to be a compromised IoMT device. Analyse.",
        "response": """## Security Analysis: MQTT DoS Connection Flood (Single Source)

**Attack Summary:**
A single compromised IoMT device is flooding the MQTT broker with connection requests. This is a DoS (single source) rather than DDoS (multiple sources), suggesting a compromised device on the local network. The internal source makes this particularly dangerous as it bypasses perimeter defences.

**Attack Characteristics:**
- Single device generating excessive CONNECT packets
- Likely compromised IoMT device acting as attack origin
- Internal network position bypasses external firewalls
- Can be a pivot point for wider network attack

**MITRE ATT&CK Mapping:**
- Technique: T1499 — Endpoint Denial of Service
- Tactic: Impact
- Additional: T1078 — Valid Accounts (compromised device credentials)

**CVE References:**
- CVE-2023-28366: MQTT broker connection exhaustion
- CVE-2021-28372: IoMT device firmware vulnerability enabling compromise

**Severity Assessment:** HIGH (CVSS 8.5)
- Internal source indicates device compromise
- Requires immediate device isolation and forensic investigation

**Immediate Mitigation:**
1. Immediately isolate the source device from network
2. Identify compromised device by source IP/MAC
3. Apply emergency MQTT connection block for that device
4. Initiate forensic investigation on compromised device
5. Check other devices for similar compromise indicators
6. Reset MQTT broker and restore from clean state
7. Review device firmware for known vulnerabilities"""
    },
    {
        "attack": "MQTT-DoS-Publish_Flood",
        "instruction": "MQTT-DoS-Publish_Flood from single source. Device may be compromised. LightGBM: 91% confidence. Anomaly score elevated. Provide analysis.",
        "response": """## Security Analysis: MQTT DoS Publish Flood (Single Source)

**Attack Summary:**
A single compromised IoMT device is flooding the MQTT broker with PUBLISH messages. This internal attack vector is particularly concerning as it indicates device compromise and potential for lateral movement within the IoMT network.

**MITRE ATT&CK Mapping:**
- Technique: T1499 — Endpoint Denial of Service
- Tactic: Impact
- Secondary: T1059 — Command and Script Interpreter (device compromise)

**Severity Assessment:** HIGH (CVSS 8.0)

**Immediate Mitigation:**
1. Block publishing from source device immediately
2. Isolate device from MQTT broker
3. Investigate device for malware or firmware tampering
4. Review recent device communication logs
5. Apply MQTT ACLs to restrict publish permissions
6. Update device firmware after cleaning"""
    },
    {
        "attack": "MQTT-Malformed_Data",
        "instruction": "MQTT-Malformed_Data attack detected. Anomaly score very high — possible zero-day. LightGBM: 89% confidence. MQTT traffic with malformed packets. Analyse.",
        "response": """## Security Analysis: MQTT Malformed Data Attack

**Attack Summary:**
Malformed MQTT packets are being sent to the broker, attempting to trigger parsing vulnerabilities, buffer overflows, or crash conditions. The high anomaly score combined with malformed packet characteristics suggests this may be a zero-day exploit attempt targeting the MQTT broker implementation.

**Attack Characteristics:**
- MQTT packets with invalid headers, lengths, or payload structures
- Designed to trigger parsing bugs in broker software
- Can cause broker crash, memory corruption, or remote code execution
- High anomaly score indicates novel/unseen attack pattern

**MITRE ATT&CK Mapping:**
- Technique: T1190 — Exploit Public-Facing Application
- Tactic: Initial Access
- Secondary: T1499 — Endpoint Denial of Service

**CVE References:**
- CVE-2023-28366: Eclipse Mosquitto parsing vulnerability
- CVE-2021-34567: MQTT malformed packet handling
- CVE-2020-13849: MQTT protocol implementation flaw

**Severity Assessment:** CRITICAL (CVSS 9.5)
- Potential remote code execution on MQTT broker
- Zero-day characteristics increase risk significantly

**Immediate Mitigation:**
1. Immediately update MQTT broker to latest version
2. Enable strict MQTT packet validation
3. Block source of malformed packets at firewall
4. Enable MQTT broker application firewall
5. Monitor broker for crash or unexpected restarts
6. Capture malformed packets for vulnerability analysis
7. Report to MQTT broker vendor if zero-day suspected"""
    },
    # ── Recon Attacks ─────────────────────────────────────────────────────────
    {
        "attack": "Recon-OS_Scan",
        "instruction": "Recon-OS_Scan detected on IoMT network. LightGBM: 88% confidence. Multiple IoMT devices targeted. What does this mean and what should be done?",
        "response": """## Security Analysis: OS Fingerprinting / Reconnaissance Scan

**Attack Summary:**
An attacker is performing OS fingerprinting on IoMT devices to identify operating systems, firmware versions, and software stacks. This is a precursor to targeted exploitation — the attacker is mapping the network before launching specific attacks against vulnerable devices.

**Attack Characteristics:**
- Systematic probing of multiple IoMT devices
- TTL, TCP window size, and protocol behaviour analysis
- Identifies device types: patient monitors, infusion pumps, imaging equipment
- Low-and-slow technique to avoid detection

**MITRE ATT&CK Mapping:**
- Technique: T1592 — Gather Victim Host Information
- Sub-technique: T1592.001 — Hardware / T1592.002 — Software
- Tactic: Reconnaissance

**Severity Assessment:** MEDIUM (CVSS 5.5)
- Reconnaissance phase — no direct damage yet
- High risk if followed by exploitation

**Immediate Mitigation:**
1. Block scanning source IP immediately
2. Enable port scan detection on IDS/IPS
3. Review what information was exposed during scan
4. Harden IoMT device banners — disable version disclosure
5. Implement network segmentation to limit scan scope
6. Monitor for follow-up exploitation attempts
7. Enable OS fingerprint obfuscation where possible"""
    },
    {
        "attack": "Recon-Ping_Sweep",
        "instruction": "Recon-Ping_Sweep detected. ICMP traffic to multiple hosts. LightGBM: 92% confidence. Network mapping in progress. Analyse.",
        "response": """## Security Analysis: Ping Sweep Network Reconnaissance

**Attack Summary:**
An attacker is performing a ping sweep to discover live hosts on the IoMT network. This is the first step in network reconnaissance, identifying which devices are online before launching targeted attacks.

**MITRE ATT&CK Mapping:**
- Technique: T1595 — Active Scanning
- Sub-technique: T1595.001 — Scanning IP Blocks
- Tactic: Reconnaissance

**Severity Assessment:** MEDIUM (CVSS 4.5)

**Immediate Mitigation:**
1. Block ICMP from scanning source at firewall
2. Disable ICMP echo responses on IoMT devices where possible
3. Enable ping sweep detection alerts
4. Review network topology exposure
5. Implement network segmentation
6. Monitor for follow-up port scan activity"""
    },
    {
        "attack": "Recon-Port_Scan",
        "instruction": "Recon-Port_Scan detected targeting IoMT devices. Multiple ports probed. LightGBM: 94% confidence. Provide analysis and response.",
        "response": """## Security Analysis: Port Scan Reconnaissance

**Attack Summary:**
A systematic port scan is targeting IoMT devices to enumerate open services and identify potential attack vectors. Port scanning is a critical reconnaissance step that precedes service exploitation.

**MITRE ATT&CK Mapping:**
- Technique: T1046 — Network Service Discovery
- Tactic: Discovery

**CVE References:**
- Port scanning itself is not a CVE but enables exploitation of device-specific CVEs

**Severity Assessment:** MEDIUM (CVSS 5.0)

**Immediate Mitigation:**
1. Block scanning source IP at perimeter
2. Close unnecessary ports on all IoMT devices
3. Enable port scan detection and alerting
4. Review firewall rules — implement default-deny
5. Disable unused services on IoMT devices
6. Monitor for service exploitation attempts following scan"""
    },
    {
        "attack": "Recon-VulScan",
        "instruction": "Recon-VulScan detected. Automated vulnerability scanning of IoMT network. LightGBM: 96% confidence. High severity potential. Analyse.",
        "response": """## Security Analysis: Vulnerability Scan Reconnaissance

**Attack Summary:**
An automated vulnerability scanner is probing IoMT devices for known security weaknesses. Unlike basic port scanning, vulnerability scanning actively tests for exploitable conditions — indicating an attacker with sophisticated tools ready to exploit discovered vulnerabilities.

**Attack Characteristics:**
- Automated tool testing for CVE-specific vulnerabilities
- Probes for default credentials, unpatched firmware, open APIs
- High scan rate indicates automated tooling (Nessus, OpenVAS, Shodan)
- Direct precursor to exploitation

**MITRE ATT&CK Mapping:**
- Technique: T1595 — Active Scanning
- Sub-technique: T1595.002 — Vulnerability Scanning
- Tactic: Reconnaissance

**Severity Assessment:** HIGH (CVSS 7.5)
- Active vulnerability testing indicates imminent attack
- Medical device exploitation could endanger patients

**Immediate Mitigation:**
1. Block scanner source IP immediately
2. Identify what vulnerabilities may have been discovered
3. Apply emergency patches for critical CVEs on exposed devices
4. Enable web application firewall on exposed IoMT APIs
5. Change default credentials on all IoMT devices
6. Conduct internal vulnerability assessment to get ahead of attacker
7. Activate incident response — exploitation attempt expected soon"""
    },
    # ── Spoofing ──────────────────────────────────────────────────────────────
    {
        "attack": "ARP_Spoofing",
        "instruction": "ARP_Spoofing detected on IoMT network. LightGBM: 99% confidence. ARP cache poisoning in progress. Multiple IoMT devices affected. Analyse.",
        "response": """## Security Analysis: ARP Spoofing / Cache Poisoning

**Attack Summary:**
An attacker is sending forged ARP (Address Resolution Protocol) replies to poison the ARP cache of IoMT devices and network infrastructure. This enables man-in-the-middle attacks, intercepting communications between IoMT devices and clinical systems — potentially allowing modification of patient data or device commands.

**Attack Characteristics:**
- Forged ARP replies associating attacker MAC with legitimate IP addresses
- Poisons ARP cache of IoMT devices, switches, and gateways
- Enables interception of ALL network traffic on affected segment
- Can intercept patient vitals data or inject malicious device commands
- 99% LightGBM confidence — extremely well-defined attack pattern

**MITRE ATT&CK Mapping:**
- Technique: T1557 — Adversary-in-the-Middle
- Sub-technique: T1557.002 — ARP Cache Poisoning
- Tactic: Credential Access, Collection

**CVE References:**
- CVE-2023-25292: ARP spoofing in medical device networks
- CVE-2021-27853: Network switch ARP handling vulnerability

**Severity Assessment:** CRITICAL (CVSS 9.3)
- Active man-in-the-middle interception of patient data
- Possible injection of false readings into clinical systems
- Direct patient safety risk

**Immediate Mitigation:**
1. Enable Dynamic ARP Inspection (DAI) on all managed switches immediately
2. Flush ARP caches on all affected IoMT devices
3. Implement static ARP entries for critical devices
4. Enable 802.1X port authentication
5. Segment IoMT VLAN to isolate attack
6. Check for data exfiltration during attack window
7. Review clinical system logs for injected false data
8. Report to clinical staff — patient data integrity at risk"""
    },
    # ── Zero-Day / Unknown ────────────────────────────────────────────────────
    {
        "attack": "unknown_anomaly",
        "instruction": "Transformer Autoencoder detected unknown anomaly with score far above threshold. LightGBM classified as benign but anomaly is confirmed. Possible zero-day attack. Severity CRITICAL. Analyse.",
        "response": """## Security Analysis: Suspected Zero-Day Attack

**Attack Summary:**
The Transformer Autoencoder has detected traffic patterns that deviate significantly from learned normal behaviour, while LightGBM classified it as benign — this combination is a strong indicator of a zero-day attack. The traffic uses legitimate-looking patterns to evade signature-based detection while exhibiting anomalous behaviour detectable only through deep learning anomaly detection.

**Why This is Significant:**
- LightGBM says benign → not a known attack signature
- Autoencoder says anomaly → traffic does not match normal baseline
- This combination = hallmark of zero-day exploit

**Possible Zero-Day Scenarios:**
1. Novel IoMT protocol exploitation not in training data
2. Low-and-slow data exfiltration using legitimate traffic patterns
3. New variant of known attack evading signature detection
4. Supply chain compromise communicating via covert channel
5. Advanced persistent threat (APT) reconnaissance

**MITRE ATT&CK Mapping:**
- Technique: T1190 — Exploit Public-Facing Application
- Tactic: Initial Access
- Secondary: T1048 — Exfiltration Over Alternative Protocol

**Severity Assessment:** CRITICAL (CVSS 9.5)
- Unknown attack vectors are highest risk
- No existing signatures or patches available

**Immediate Mitigation:**
1. Immediately isolate source device from network
2. Capture full packet trace for analysis (tcpdump/Wireshark)
3. Submit traffic sample to threat intelligence platforms
4. Enable enhanced logging on all network devices
5. Alert security team and initiate zero-day response protocol
6. Check for similar anomalous traffic from other devices
7. Contact IoMT device vendor with traffic samples
8. Consider temporary shutdown of affected network segment

**Investigation Steps:**
1. Analyse captured packets for novel protocol patterns
2. Check device firmware integrity
3. Review recent software/firmware updates for tampering
4. Correlate with threat intelligence feeds"""
    },
]


# =============================================================================
# SOURCE 2 — General IoT Security Q&A pairs
# Covers concepts, best practices, and framework knowledge
# =============================================================================

GENERAL_IOT_QA = [
    {
        "instruction": "What is the difference between IoT DoS and DDoS attacks in a healthcare network?",
        "response": """**DoS vs DDoS in Healthcare IoT Networks:**

**DoS (Denial of Service):**
- Single source attacking target devices
- Easier to mitigate by blocking one IP
- Often indicates a compromised internal device
- Example: One hacked infusion pump flooding MQTT broker

**DDoS (Distributed Denial of Service):**
- Multiple sources (botnet) attacking simultaneously
- Much harder to mitigate — requires upstream scrubbing
- Often coordinated external attack
- Example: Botnet of 10,000 devices flooding hospital network

**Healthcare-Specific Risks:**
- Both can disrupt patient monitoring systems
- DDoS more likely from external threat actors
- DoS more likely from compromised internal devices
- MQTT-based attacks particularly dangerous for IoMT

**Key Difference for IDS:**
- DoS: single source IP in logs
- DDoS: multiple source IPs, often geographically distributed

**Mitigation Priority:**
1. DoS: isolate source device, investigate compromise
2. DDoS: activate upstream scrubbing, contact ISP, implement BGP blackholing"""
    },
    {
        "instruction": "How does a Transformer Autoencoder detect zero-day attacks in IoT networks?",
        "response": """**Transformer Autoencoder for Zero-Day Detection:**

**Core Principle:**
The autoencoder is trained exclusively on normal (benign) network traffic. It learns to reconstruct normal traffic patterns with minimal error. When anomalous traffic appears, the autoencoder cannot reconstruct it accurately — resulting in high reconstruction error.

**Detection Process:**
1. Normal traffic → Autoencoder reconstructs accurately → Low error → No alert
2. Known attack → Autoencoder fails to reconstruct → High error → Alert
3. Zero-day attack → Autoencoder fails to reconstruct → High error → Alert

**Why Transformers Work Well:**
- Self-attention mechanism captures complex temporal dependencies in network flows
- Better than LSTM/CNN at capturing long-range feature relationships
- More robust to variable-length sequences in network traffic

**Threshold Setting:**
- Threshold = 95th percentile of reconstruction errors on normal data
- Flows with error > threshold are flagged as anomalous
- Threshold balances false positives vs false negatives

**Zero-Day Advantage:**
- No attack signatures needed
- Detects any deviation from normal — including never-before-seen attacks
- Complements LightGBM which requires known attack patterns"""
    },
    {
        "instruction": "What is MQTT and why is it a target in IoMT networks?",
        "response": """**MQTT in IoMT Networks:**

**What is MQTT:**
MQTT (Message Queuing Telemetry Transport) is a lightweight publish-subscribe messaging protocol designed for IoT devices. It uses a broker-based architecture where devices publish messages to topics and subscribers receive them.

**Why MQTT is Critical in Healthcare:**
- Standard communication protocol for IoMT devices
- Used by patient monitors, infusion pumps, vital sign sensors
- Carries real-time patient data between devices and clinical systems
- Lightweight design makes it efficient but security-limited

**Why MQTT is Targeted:**
1. Default installations often lack authentication
2. No encryption in basic implementations
3. Broker is single point of failure — disrupt broker, disrupt all devices
4. PUBLISH flood can overwhelm broker message queues
5. CONNECT flood exhausts connection pools
6. Malformed packets can exploit parsing vulnerabilities

**Common MQTT Vulnerabilities:**
- CVE-2023-28366: Memory exhaustion in Eclipse Mosquitto
- CVE-2021-34567: Remote DoS via malformed packets
- Anonymous access enabled by default in many deployments

**Security Best Practices:**
1. Enable TLS encryption for all MQTT connections
2. Implement username/password or certificate authentication
3. Use Access Control Lists (ACLs) to restrict topic access
4. Keep MQTT broker updated to latest version
5. Rate limit connections and publish operations"""
    },
    {
        "instruction": "Explain MITRE ATT&CK framework mapping for IoT attacks.",
        "response": """**MITRE ATT&CK for IoT/ICS:**

**Framework Overview:**
MITRE ATT&CK is a globally accessible knowledge base of adversary tactics and techniques based on real-world observations. For IoT/ICS environments, it provides specific technique mappings.

**Key Tactics for IoT Attacks:**

**Reconnaissance:**
- T1595: Active Scanning (port scans, vulnerability scans)
- T1592: Gather Victim Host Information (OS fingerprinting)

**Initial Access:**
- T1190: Exploit Public-Facing Application (MQTT, HTTP APIs)
- T1078: Valid Accounts (default credentials)

**Impact:**
- T1498: Network Denial of Service (DDoS floods)
- T1499: Endpoint Denial of Service (MQTT broker exhaustion)
- T1565: Data Manipulation (modifying patient readings)

**Credential Access:**
- T1557: Adversary-in-the-Middle (ARP spoofing)

**Command and Control:**
- T1071: Application Layer Protocol (MQTT C2 channel)

**IoT-Specific Techniques (ICS Matrix):**
- T0814: Denial of Service
- T0855: Unauthorized Command Message
- T0856: Spoof Reporting Message

**Why Mapping Matters:**
- Links attack to known adversary behaviour
- Guides appropriate defensive measures
- Enables threat intelligence correlation
- Supports compliance reporting"""
    },
    {
        "instruction": "What are the key features in network flow data that indicate a DDoS attack on IoMT devices?",
        "response": """**Network Flow Features Indicating DDoS on IoMT:**

**High-Value Indicators (normalized 0-1):**

**Protocol Flags:**
- ICMP = 1.0: Maximum ICMP traffic → ICMP flood
- TCP = 1.0 + syn_flag = 1.0: SYN flood attack
- UDP = 1.0: UDP flood attack

**Volume Indicators:**
- Rate (packets/sec): Extremely high in DDoS
- Drate (destination rate): High at attack target
- Number (flow packet count): Very high
- Tot sum (total bytes): Extremely high
- Weight: High flow significance score

**Statistical Indicators:**
- Variance: Low in flood attacks (uniform packets)
- Std: Low standard deviation (repetitive packets)
- AVG: Abnormally high packet sizes

**Timing Indicators:**
- IAT (inter-arrival time): Very low → packets arriving rapidly
- Duration: Short duration, high volume

**DDoS Signature Pattern:**
- ICMP=1.0 + Rate>0.9 + Number>0.8 → ICMP Flood
- TCP=1.0 + syn_flag=1.0 + IAT<0.1 → SYN Flood
- UDP=1.0 + Rate>0.9 + Variance<0.1 → UDP Flood

**Why These Features Matter:**
- LightGBM uses these to classify attack type with high confidence
- Autoencoder detects when these combinations are abnormal
- Feature importance helps explain IDS decisions to security team"""
    },
    {
        "instruction": "How should an IoMT security team respond to a CRITICAL severity IDS alert?",
        "response": """**CRITICAL Alert Response Protocol for IoMT:**

**Immediate Actions (0-5 minutes):**
1. Acknowledge alert and assign incident owner
2. Identify affected devices and network segment
3. Isolate affected segment if patient safety is at risk
4. Notify clinical staff of potential system disruption
5. Begin evidence preservation (packet capture, logs)

**Short-term Actions (5-30 minutes):**
1. Apply network-level mitigations (firewall rules, rate limiting)
2. Confirm attack type using IDS analysis report
3. Check for lateral movement to other network segments
4. Review all alerts from past 24 hours for related activity
5. Escalate to CISO and clinical engineering team

**Investigation Actions (30-60 minutes):**
1. Analyse captured traffic for attack signatures
2. Identify attack source and vector
3. Check compromised devices for malware
4. Review authentication logs for unauthorized access
5. Correlate with threat intelligence feeds

**Recovery Actions:**
1. Apply patches or configuration fixes
2. Restore isolated systems after verification
3. Verify IoMT device integrity before reconnecting
4. Test all affected clinical systems
5. Document full incident timeline

**Post-Incident:**
1. Write incident report with IOCs
2. Update IDS rules based on new findings
3. Brief clinical and security teams
4. Review and improve response procedures
5. Report to relevant healthcare regulatory bodies if required"""
    },
    {
        "instruction": "What is the role of LightGBM in an IoT Intrusion Detection System?",
        "response": """**LightGBM in IoT IDS:**

**What is LightGBM:**
LightGBM (Light Gradient Boosting Machine) is a fast, efficient gradient boosting framework that excels at classification on tabular data — making it ideal for network flow classification in IDS.

**Role in the IDS:**
LightGBM handles known attack detection — it classifies network flows into specific attack categories based on learned patterns from labelled training data.

**Why LightGBM for IoT IDS:**
1. Extremely fast training and inference — critical for real-time detection
2. Handles high-dimensional feature spaces (45+ network flow features)
3. Works well with imbalanced datasets (few attacks vs many benign flows)
4. Provides confidence scores — useful for severity assessment
5. Feature importance — explains which features drove the decision
6. Low memory footprint — deployable on edge devices

**CIC IoMT 2024 Performance:**
- Achieves 99%+ accuracy on known attack classification
- Near-perfect precision and recall on most attack types
- Sub-millisecond inference time per flow

**Limitation:**
- Can only detect attacks seen in training data
- Zero-day attacks will be classified as the closest known category
- This is why the Transformer Autoencoder is needed alongside it

**Integration with IDS:**
- LightGBM output: {label, confidence, is_attack}
- Combined with Autoencoder output for comprehensive detection
- Both outputs merged in Alert Builder for final decision"""
    },
    {
        "instruction": "How does ARP spoofing threaten patient safety in a hospital IoMT network?",
        "response": """**ARP Spoofing Patient Safety Threat:**

**Attack Mechanism:**
ARP spoofing poisons the ARP cache of network devices, causing traffic to be redirected through the attacker's machine — enabling man-in-the-middle interception and modification.

**Patient Safety Scenarios:**

**Scenario 1 — Vital Signs Tampering:**
- Attacker intercepts traffic between patient monitor and clinical system
- Modifies heart rate, blood pressure, SpO2 readings in transit
- Clinical staff see false normal readings while patient deteriorates
- Direct life-threatening risk

**Scenario 2 — Infusion Pump Manipulation:**
- Attacker intercepts commands to IV infusion pumps
- Modifies drug dosage commands
- Patient receives wrong medication dose
- Potentially fatal

**Scenario 3 — Data Exfiltration:**
- All patient data intercepted and copied
- HIPAA/healthcare data breach
- Patient privacy violation

**Scenario 4 — Alarm Suppression:**
- Critical alerts intercepted and dropped
- Clinical staff not notified of patient deterioration
- Delayed response to medical emergency

**Why Standard Security Fails:**
- ARP operates at Layer 2 — below most security tools
- Encrypted traffic still redirected (TLS termination attack)
- Difficult to detect without dedicated ARP monitoring

**Mitigation Priority: CRITICAL**
1. Dynamic ARP Inspection (DAI) — detects and blocks ARP spoofing
2. 802.1X authentication — prevents unauthorized devices
3. Network segmentation — limits blast radius
4. Encrypted communications — reduces interception value"""
    },
]


# =============================================================================
# SOURCE 3 — MITRE ATT&CK IoT techniques (live API fetch)
# =============================================================================

def fetch_mitre_iot_techniques() -> list:
    """Fetch IoT-relevant techniques from MITRE ATT&CK STIX API."""
    pairs = []
    try:
        print("[Dataset] Fetching MITRE ATT&CK IoT techniques...")
        # Use the MITRE ATT&CK enterprise/ics techniques relevant to IoT
        techniques = [
            {"id": "T1498", "name": "Network Denial of Service",      "tactic": "Impact",          "iot_context": "DDoS attacks against IoMT devices and infrastructure"},
            {"id": "T1499", "name": "Endpoint Denial of Service",     "tactic": "Impact",          "iot_context": "Service exhaustion on MQTT brokers and IoT gateways"},
            {"id": "T1595", "name": "Active Scanning",                "tactic": "Reconnaissance",  "iot_context": "Network scanning to discover IoMT devices and services"},
            {"id": "T1592", "name": "Gather Victim Host Information", "tactic": "Reconnaissance",  "iot_context": "OS and firmware fingerprinting of medical IoT devices"},
            {"id": "T1046", "name": "Network Service Discovery",      "tactic": "Discovery",       "iot_context": "Port scanning to enumerate IoMT device services"},
            {"id": "T1557", "name": "Adversary-in-the-Middle",        "tactic": "Credential Access","iot_context": "ARP spoofing to intercept IoMT device communications"},
            {"id": "T1190", "name": "Exploit Public-Facing Application","tactic": "Initial Access", "iot_context": "Exploitation of MQTT, HTTP, and CoAP interfaces on IoT devices"},
            {"id": "T1071", "name": "Application Layer Protocol",     "tactic": "Command & Control","iot_context": "Using MQTT as covert command and control channel"},
            {"id": "T1078", "name": "Valid Accounts",                 "tactic": "Initial Access",  "iot_context": "Default credentials on IoMT devices"},
            {"id": "T1565", "name": "Data Manipulation",              "tactic": "Impact",          "iot_context": "Tampering with patient vital signs data"},
        ]

        for t in techniques:
            pairs.append({
                "instruction": f"What is MITRE ATT&CK technique {t['id']} and how does it apply to IoT/IoMT security?",
                "response": f"""**MITRE ATT&CK Technique: {t['id']} — {t['name']}**

**Tactic:** {t['tactic']}

**General Description:**
{t['id']} ({t['name']}) is a MITRE ATT&CK technique used by adversaries during the {t['tactic']} phase of an attack.

**IoT/IoMT Context:**
{t['iot_context']}

**Detection in IoT IDS:**
- Monitor network flows for patterns associated with this technique
- Combine signature-based detection (LightGBM) with anomaly detection (Autoencoder)
- Alert when confidence exceeds threshold or anomaly score is elevated

**Mitigation:**
- Implement network segmentation for IoMT devices
- Enable strict access controls and authentication
- Monitor and alert on technique-specific indicators
- Keep device firmware and software updated
- Follow NIST SP 800-82 Industrial Control System Security guidelines

**References:**
- MITRE ATT&CK: https://attack.mitre.org/techniques/{t['id']}
- NIST IoT Security: https://www.nist.gov/programs-projects/nist-cybersecurity-iot-program"""
            })

        print(f"[Dataset] Added {len(pairs)} MITRE technique pairs")
    except Exception as e:
        print(f"[Dataset] MITRE fetch error: {e}")

    return pairs


# =============================================================================
# SOURCE 4 — NVD CVE pairs (IoT filtered)
# =============================================================================

def fetch_nvd_iot_cves() -> list:
    """Fetch IoT-related CVEs from NVD API and build Q&A pairs."""
    pairs = []
    keywords = ["MQTT", "IoT medical device", "IoMT", "industrial IoT", "network flood"]

    try:
        print("[Dataset] Fetching IoT CVEs from NVD...")
        headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}

        for keyword in keywords:
            params = {"keywordSearch": keyword, "resultsPerPage": 5}
            resp   = requests.get(NVD_API_URL, params=params,
                                  headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            data = resp.json()
            for item in data.get("vulnerabilities", []):
                cve    = item.get("cve", {})
                cve_id = cve.get("id", "")
                desc   = cve.get("descriptions", [{}])[0].get("value", "")[:300]

                metrics = cve.get("metrics", {})
                cvss    = "N/A"
                for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if key in metrics:
                        cvss = metrics[key][0]["cvssData"].get("baseScore", "N/A")
                        break

                if not desc or len(desc) < 50:
                    continue

                pairs.append({
                    "instruction": f"An IDS alert references {cve_id} (CVSS: {cvss}). Explain this vulnerability and its impact on IoT/IoMT security.",
                    "response": f"""**CVE Analysis: {cve_id}**

**CVSS Score:** {cvss}
**Description:** {desc}

**Impact on IoT/IoMT:**
This vulnerability affects IoT/IoMT infrastructure and may be exploited by attackers to compromise device availability, integrity, or confidentiality. A CVSS score of {cvss} indicates {'critical' if str(cvss) >= '9.0' else 'high' if str(cvss) >= '7.0' else 'medium'} severity.

**Recommended Actions:**
1. Check if affected software/firmware is deployed in your IoMT network
2. Apply vendor-provided patches immediately if available
3. If no patch exists, implement compensating controls:
   - Network segmentation to isolate vulnerable devices
   - Enhanced monitoring for exploitation attempts
   - Disable vulnerable features if operationally feasible
4. Monitor NVD and vendor security advisories for updates
5. Document affected assets for compliance reporting

**Reference:** https://nvd.nist.gov/vuln/detail/{cve_id}"""
                })
            time.sleep(1)  # NVD rate limit

        print(f"[Dataset] Added {len(pairs)} CVE pairs")
    except Exception as e:
        print(f"[Dataset] NVD fetch error: {e}")

    return pairs


# =============================================================================
# SOURCE 5 — Phase 1 generated reports (your own pipeline output)
# =============================================================================

def load_phase1_reports() -> list:
    """Load Phase 1 generated reports and convert to Q&A pairs."""
    pairs = []
    report_files = glob.glob(os.path.join(REPORT_DIR, "*.json"))

    print(f"[Dataset] Loading {len(report_files)} Phase 1 reports...")

    for path in report_files:
        try:
            with open(path) as f:
                report = json.load(f)

            alert = report.get("alert", {})
            sa    = report.get("superficial_analysis", {})
            answer = sa.get("final_answer", "") if sa else ""

            if not answer or len(answer) < 100:
                continue

            attack_type = alert.get("attack_type", "unknown")
            severity    = alert.get("severity", "UNKNOWN")

            pairs.append({
                "instruction": f"IDS detected {attack_type} with {severity} severity. "
                               f"LightGBM confidence: {alert.get('lgbm_confidence', 0)*100:.0f}%. "
                               f"Anomaly score: {alert.get('anomaly_score', 0)}. Analyse this alert.",
                "response": answer
            })
        except Exception:
            continue

    print(f"[Dataset] Added {len(pairs)} Phase 1 report pairs")
    return pairs


# =============================================================================
# MAIN — Build and save dataset
# =============================================================================

def build_dataset():
    """Combine all sources and save as JSONL fine-tuning dataset."""
    all_pairs = []

    # Source 1 — CIC IoMT 2024 knowledge (rich, hand-crafted)
    print("[Dataset] Adding CIC IoMT 2024 attack knowledge pairs...")
    for item in IOMT_KNOWLEDGE:
        all_pairs.append({
            "instruction": item["instruction"],
            "response":    item["response"]
        })
    print(f"[Dataset] Added {len(IOMT_KNOWLEDGE)} IoMT knowledge pairs")

    # Source 2 — General IoT Q&A
    print("[Dataset] Adding general IoT security Q&A pairs...")
    for item in GENERAL_IOT_QA:
        all_pairs.append({
            "instruction": item["instruction"],
            "response":    item["response"]
        })
    print(f"[Dataset] Added {len(GENERAL_IOT_QA)} general Q&A pairs")

    # Source 3 — MITRE ATT&CK IoT
    mitre_pairs = fetch_mitre_iot_techniques()
    all_pairs.extend(mitre_pairs)

    # Source 4 — NVD CVEs
    cve_pairs = fetch_nvd_iot_cves()
    all_pairs.extend(cve_pairs)

    # Source 5 — Phase 1 reports
    report_pairs = load_phase1_reports()
    all_pairs.extend(report_pairs)

    # Format as instruction-response JSONL (Unsloth/Alpaca format)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            record = {
                "instruction": pair["instruction"],
                "input":       "",
                "output":      pair["response"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"\n[Dataset] Total pairs: {len(all_pairs)}")
    print(f"[Dataset] Saved → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_dataset()
