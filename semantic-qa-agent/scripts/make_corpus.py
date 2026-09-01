"""Generate the synthetic company-handbook corpus.

The corpus is written as code rather than committed prose so that it is
reproducible and so that the *vocabulary-mismatch* cases the golden set depends
on are authored deliberately. Several sections are written to state a fact using
vocabulary that a user would never type -- for example "Non-refundable Bookings"
for the question "can I get money back if a trip is cancelled?" -- because those
zero-overlap pairs are exactly what separates semantic retrieval from keyword
search, and a corpus without them cannot demonstrate the difference.
"""

from pathlib import Path

DOCS: dict[str, str] = {}

DOCS["payroll_policy.md"] = """# Payroll and Compensation Policy

## Salary Disbursement

Salaries are credited on the last working day of each calendar month. Where the
last working day falls on a bank holiday, disbursement moves to the preceding
working day. Payslips are published to the employee self-service portal within
two working days of disbursement.

## Salary Revision Cycle

The annual compensation review takes effect from 1 April each year. Employees
who joined on or after 1 January of the same year are not eligible for that
cycle and are considered in the following one. Revision letters are issued by
15 April.

## Variable Pay and Bonus

Variable pay is calculated as a percentage of fixed annual salary and is linked
to both individual rating and company performance. Payout occurs once per year
in the May payroll. An employee serving notice on the payout date is not
eligible for variable pay.

## Provident Fund and Deductions

Statutory provident fund contribution is 12 percent of basic salary, matched by
the employer. Professional tax and income tax are deducted at source. Employees
may submit investment declarations until 31 January to adjust tax deduction.

## Full and Final Settlement

Full and final settlement is processed within 45 days of the last working day,
after clearance from the reporting manager, IT asset recovery, and finance. Any
unrecovered advance or asset is deducted from the settlement amount.
"""

DOCS["procurement_policy.md"] = """# Procurement and Vendor Policy

## Purchase Authorisation Limits

Purchases up to INR 50,000 may be approved by a team manager. Purchases between
INR 50,000 and INR 5,00,000 require department head approval. Anything above
INR 5,00,000 requires a director signature and competitive quotes from at least
three vendors.

## Vendor Onboarding

New vendors must complete the vendor registration form, submit tax registration
documents, and pass a compliance screening before any purchase order is raised.
Compliance screening typically takes five working days.

## Purchase Orders

No payment will be released without a purchase order raised in advance of the
work. Retrospective purchase orders are permitted only in a documented emergency
and require director approval within 48 hours.

## Invoice Payment Terms

Standard payment terms are 30 days from invoice receipt. Early payment discounts
must be recorded on the purchase order. Disputed invoices are placed on hold and
the vendor is notified within five working days of the dispute being raised.

## Software Purchases

All software purchases require security review before the purchase order is
raised, regardless of value. Free and open-source tools used in production also
require review, since licence obligations may apply.
"""

DOCS["performance_review.md"] = """# Performance Management Framework

## Review Cadence

Formal performance reviews are held twice per year, in March and September.
Continuous feedback is expected throughout the year, and the formal review should
contain no surprises for the employee.

## Rating Scale

Employees are rated on a five-point scale: Outstanding, Exceeds Expectations,
Meets Expectations, Partially Meets, and Does Not Meet. Ratings are calibrated
across teams before release in order to remove individual manager bias.

## Goal Setting

Goals are agreed between the employee and manager at the start of each review
period and recorded in the performance portal. Each goal must have a measurable
outcome and a target date. Goals may be revised mid-cycle by mutual agreement.

## Performance Improvement Plan

An employee rated Does Not Meet is placed on a structured improvement plan of 60
to 90 days with weekly checkpoints and a named support partner. Successful
completion returns the employee to the standard review cadence.

## Promotion Criteria

Promotion requires sustained performance at the next level for at least two
consecutive review cycles, a supporting business case from the manager, and
availability of the role in the team structure.
"""

DOCS["data_retention_policy.md"] = """# Data Retention and Disposal Standard

## Retention Periods

Customer transaction records are retained for seven years to satisfy statutory
audit requirements. Employee records are retained for seven years after the end
of employment. Application and server logs are retained for 180 days. Marketing
contact data is retained for three years from last engagement.

## Deletion Requests

A verified request from a data subject to erase personal data must be actioned
within 30 calendar days. Where a statutory retention obligation prevents
deletion, the data is restricted from processing instead, and the requester is
informed of the reason and the applicable retention period.

## Backups

Backups are retained for 35 days on a rolling basis. Deletion requests are
applied to live systems immediately and to backups on natural expiry, since
selective erasure inside an encrypted backup set is not technically feasible.

## Secure Disposal

Physical media are destroyed by a certified disposal vendor, with a certificate
of destruction retained for audit. Devices holding Restricted data must be
cryptographically erased before disposal or reissue.
"""

DOCS["engineering_handbook.md"] = """# Engineering Handbook

## Code Review Standards

Every change to a production repository requires review and approval by at least
one other engineer. Reviews should be completed within one working day. A change
that alters authentication, payment, or data-retention behaviour requires a
second reviewer drawn from the security team.

## Branching and Release

Work happens on short-lived feature branches cut from main. Releases are cut
weekly on Wednesday. Hotfixes may be released at any time with approval from the
on-call lead and must be merged back to main the same day.

## Testing Expectations

New code paths require automated tests. Pull requests that reduce overall test
coverage are blocked by the pipeline. Flaky tests must be fixed or quarantined
within one week of being reported. Silently disabling a test is not acceptable.

## On-Call Rotation

The on-call rotation runs weekly, handing over on Monday morning. The on-call
engineer acknowledges a critical alert within 15 minutes and provides a written
incident summary within 24 hours of resolution.

## Production Access

Direct write access to production databases is not granted to individuals.
Changes are applied through reviewed migrations. Emergency read access may be
granted for a maximum of four hours and is logged and reviewed afterwards.
"""

DOCS["facilities_guide.md"] = """# Workplace and Facilities Guide

## Building Access

Access badges are active between 07:00 and 22:00 on weekdays. Weekend or
out-of-hours access requires a request to facilities at least one working day in
advance. A lost badge must be reported immediately so that it can be deactivated.

## Desk Booking

Desks are booked through the workplace app up to two weeks in advance. A booking
not claimed by 11:00 is released automatically to other employees. Teams may
reserve a block of desks for a fixed collaboration day each week.

## Meeting Rooms

Meeting rooms seating eight or more may not be booked for fewer than three
attendees during core hours of 10:00 to 16:00. Recurring bookings are reviewed
quarterly and cancelled where attendance is consistently below capacity.

## Visitors

Visitors must be registered in advance and are escorted at all times inside the
building. Visitors are not permitted in engineering or server areas without
written approval from the security team.

## Parking and Commute

Parking is allocated by ballot each quarter. A shuttle service runs from the
nearest metro station every 20 minutes between 08:00 and 10:30 and between 17:00
and 20:00.
"""

DOCS["customer_support_sla.md"] = """# Customer Support Service Levels

## Priority Definitions

A Priority 1 issue means the product is completely unavailable for all users of
an account. Priority 2 means a core function is unusable with no workaround.
Priority 3 covers degraded behaviour where a workaround is available. Priority 4
is a question or a cosmetic issue.

## Response and Resolution Targets

Priority 1 receives a first response within 30 minutes and a target resolution of
four hours. Priority 2 receives a response within two hours and resolution within
one business day. Priority 3 responds within one business day. Priority 4
responds within three business days.

## Escalation Path

A customer may request escalation at any time. Escalated tickets are reviewed by
the support lead within one hour during business hours. Any Priority 1 issue open
for more than two hours is escalated automatically to engineering.

## Maintenance Windows

Planned maintenance occurs on the first Sunday of each month between 02:00 and
06:00 in the regional time zone of the customer. Customers receive notice at
least seven days in advance. Emergency maintenance may occur with two hours of
notice.

## Refunds and Service Credits

Where monthly uptime falls below 99.5 percent, affected accounts receive a
service credit of ten percent of the monthly fee. Credits are applied
automatically and do not require the customer to raise a claim.
"""

DOCS["training_policy.md"] = """# Learning and Development Policy

## Annual Learning Budget

Each employee has an annual learning budget of INR 40,000 for courses,
certifications, conferences and books. The budget runs on the financial year and
does not carry over. Manager approval is required before booking.

## Certification Reimbursement

The cost of a professional certification is reimbursed on successful completion.
A failed attempt is reimbursed once, and subsequent attempts are at the expense
of the employee. Certifications must be relevant to the current or next role.

## Conference Attendance

Attendance at one external conference per year is supported, including travel
booked under the standard travel policy. Attendees are expected to share a short
written summary with their team within two weeks of returning.

## Internal Tech Talks

Tech talks are held every second Thursday. Any employee may propose a session.
Sessions are recorded and published to the internal library unless the speaker
opts out.

## Study Leave

Up to three days of paid study leave per year may be taken for an approved
certification examination. Study leave is separate from annual leave and does not
reduce the annual entitlement.
"""

DOCS["device_and_asset_policy.md"] = """# Device and Asset Policy

## Standard Issue

Every employee receives a laptop, a charger and a headset on joining. Engineering
roles may request an external monitor and keyboard. Hardware requests beyond the
standard issue require manager approval and are fulfilled within ten working days.

## Refresh Cycle

Laptops are refreshed every three years. An earlier replacement requires a
diagnostic report from the hardware team confirming a fault that cannot be
economically repaired.

## Damage and Loss

Accidental damage is covered by the company. A device lost outside the office
must be reported to both IT and security within 24 hours so that it can be
remotely wiped. Repeated loss may result in the employee bearing part of the
replacement cost.

## Personal Use

Limited personal use of a company device is acceptable. Installing unlicensed
software, disabling the endpoint agent, or using the device for the work of
another employer is prohibited.

## Return on Exit

All assets must be returned on or before the last working day. Unreturned assets
are recovered from the full and final settlement at their depreciated value.
"""

DOCS["communication_guidelines.md"] = """# Communication and Meetings Guidelines

## Default to Written

Decisions are recorded in writing, in the relevant project channel or document,
so that people in other time zones and people who join later can follow the
reasoning without having attended a meeting.

## Meeting Hygiene

Every meeting requires an agenda circulated in advance and a named owner. A
meeting without an agenda may be declined without explanation. Notes and actions
are posted within one working day.

## Response Time Expectations

Messages in a team channel carry no expectation of an immediate reply. A response
within one working day is the norm. Anything genuinely urgent should use the
on-call escalation path rather than a channel message.

## Focus Time

Wednesday afternoons are protected as company-wide focus time. Internal meetings
should not be scheduled then, other than incident response.

## External Communication

Public statements about the company, including social media posts about products
or customers, must be reviewed by the communications team. Employees should not
respond directly to press enquiries.
"""


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, body in DOCS.items():
        (out / name).write_text(body, encoding="utf-8")
    print(f"wrote {len(DOCS)} document(s) -> {out}")


if __name__ == "__main__":
    main()
