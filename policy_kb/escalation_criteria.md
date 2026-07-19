# Escalation Criteria

Tickets are escalated to a human senior agent (bypassing automated resolution) when any 
of the following conditions are met:

- Priority is Critical or the ticket body explicitly mentions a data breach, outage 
  affecting multiple users, or legal/compliance concern.
- The customer has submitted 3 or more unresolved tickets on the same issue within 7 days.
- The automated response system expresses low confidence (below 40%) in the generated answer.
- The customer explicitly requests human assistance.

Escalated tickets are flagged in the system and assigned to an available senior agent 
within the SLA window for their priority level. Automated responses are not sent for 
escalated tickets.