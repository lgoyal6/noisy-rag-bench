"""Synthetic bank / law-firm document corpus with gold QA pairs.

Everything here is generated. No client data, no scraped documents, nothing that
could not be handed to a prospect. The point is that the *shape* is realistic:
near-duplicate boilerplate across many counterparties, so answering a question
requires distinguishing entity A's threshold from entity B's, which is exactly
what OCR noise destroys.
"""
import random

BORROWERS = [
    "Meridian Trust Holdings", "Calderwood Financial Group", "Northfield Bancorp",
    "Ashgrove Capital Partners", "Pelham Ridge Industries", "Kestrel Marine Logistics",
    "Thornbury Retail Group", "Vanmark Aerospace", "Silverline Utilities",
    "Harrowgate Chemical", "Bexley Pharmaceuticals", "Lindmoor Data Centers",
]
FIRMS = [
    "Ashford & Vance LLP", "Coleridge Stanton", "Delahaye Whitmore LLP",
    "Farrington Blake", "Grantley Hume LLP", "Ravenscroft Piper",
]
AGENTS = ["Bank of the Republic, N.A.", "Continental First Bank", "Harborline Trust Company"]

SECTION_TEMPLATES = [
    # (title, body-with-{fields}, fact-key, question)
    ("Section 6.11 Financial Covenants",
     "The Borrower shall not permit the Consolidated Fixed Charge Coverage Ratio, "
     "measured as of the last day of any fiscal quarter, to be less than {fccr} to 1.00. "
     "The Borrower shall not permit the Consolidated Total Net Leverage Ratio as of the "
     "last day of any fiscal quarter to exceed {leverage} to 1.00. Compliance with this "
     "Section shall be evidenced by the Compliance Certificate delivered pursuant to "
     "Section 5.01(c).",
     "fccr",
     "What is the minimum Consolidated Fixed Charge Coverage Ratio {ent} must maintain?"),

    ("Section 2.06 Commitment Fee",
     "The Borrower agrees to pay to the Administrative Agent for the account of each "
     "Lender a commitment fee equal to {commfee} percent per annum on the average daily "
     "unused amount of such Lender's Revolving Commitment. The commitment fee shall "
     "accrue at all times during the Availability Period and shall be payable quarterly "
     "in arrears on the last Business Day of each March, June, September and December.",
     "commfee",
     "What commitment fee rate applies to the unused revolving commitment for {ent}?"),

    ("Section 7.02 Limitation on Indebtedness",
     "The Borrower shall not, and shall not permit any Restricted Subsidiary to, create, "
     "incur, assume or suffer to exist any Indebtedness, except purchase money "
     "Indebtedness and Capitalized Lease Obligations in an aggregate principal amount not "
     "to exceed {debtcap} at any time outstanding. Any Indebtedness incurred in reliance "
     "on this clause shall be secured only by the assets financed thereby.",
     "debtcap",
     "What is the aggregate cap on purchase money indebtedness for {ent}?"),

    ("Section 5.01 Reporting Requirements",
     "The Borrower shall deliver to the Administrative Agent, within {qdays} days after "
     "the end of each of the first three fiscal quarters, a consolidated balance sheet and "
     "the related statements of income and cash flows, certified by a Financial Officer. "
     "Annual audited statements shall be delivered within {adays} days after the end of "
     "each fiscal year, accompanied by an opinion of independent public accountants.",
     "qdays",
     "Within how many days after each fiscal quarter must {ent} deliver financial statements?"),

    ("Section 9.04 Assignments and Participations",
     "No assignment shall be made to any Person unless the assigning Lender retains, or "
     "the assignee acquires, an aggregate outstanding principal amount of not less than "
     "{assignmin}, unless the Administrative Agent otherwise consents. Each assignment "
     "shall be recorded in the Register maintained by the Administrative Agent, and a "
     "processing fee of {assignfee} shall accompany each assignment.",
     "assignmin",
     "What is the minimum assignment amount under the {ent} credit agreement?"),

    ("Section 8.01 Events of Default",
     "An Event of Default shall occur if the Borrower fails to pay any principal when due, "
     "or fails to pay any interest or fee within {graceper} Business Days after the same "
     "becomes due. A cross-default shall arise upon any default in respect of other "
     "Indebtedness having an aggregate principal amount in excess of {crossdef}.",
     "graceper",
     "How many business days of grace does {ent} have on an interest payment?"),

    ("Section 3.02 Interest Rates",
     "Each Term Benchmark Loan shall bear interest at the Adjusted Term SOFR Rate plus an "
     "Applicable Margin of {margin} basis points. Each ABR Loan shall bear interest at the "
     "Alternate Base Rate plus {abrmargin} basis points. Interest on overdue amounts shall "
     "accrue at a rate {default_add} percent per annum above the otherwise applicable rate.",
     "margin",
     "What is the SOFR applicable margin in basis points for {ent}?"),
]

ENGAGEMENT_TEMPLATES = [
    ("Scope of Engagement",
     "{firm} has been retained by {ent} in connection with the matter described above. "
     "Our engagement is limited to the specific matter identified and does not extend to "
     "tax, regulatory, or securities advice unless separately agreed in writing. The "
     "engagement partner responsible for this matter is {partner}.",
     "partner",
     "Who is the engagement partner for the {ent} matter at {firm}?"),

    ("Fees and Billing Arrangements",
     "Fees for this engagement will be billed at the standard hourly rates then in effect. "
     "The current rate for partners is {prate} per hour and for associates is {arate} per "
     "hour. Paralegal time is billed at {pararate} per hour. Invoices are rendered monthly "
     "and are payable within {payterm} days of receipt.",
     "prate",
     "What is the partner hourly rate charged to {ent} by {firm}?"),

    ("Retainer and Trust Account",
     "An advance retainer of {retainer} is required before work commences and will be held "
     "in the firm's client trust account. The retainer will be applied against the final "
     "invoice. {firm} may require replenishment of the retainer if the balance falls below "
     "{replenish}.",
     "retainer",
     "What advance retainer does {firm} require from {ent}?"),

    ("Conflicts and Confidentiality",
     "{firm} has conducted a conflicts check and has identified no disqualifying conflict "
     "with respect to this engagement. All communications are subject to the "
     "attorney-client privilege. Client files will be retained for {retention} years "
     "following the conclusion of the matter, after which they may be destroyed.",
     "retention",
     "For how many years does {firm} retain the {ent} client file after the matter closes?"),
]

PARTNERS = ["Eleanor Prescott", "Nathaniel Ogunyemi", "Priya Raghunathan",
            "Douglas Farraday", "Ingrid Solberg", "Marcus Thibodeaux",
            "Yuki Tanabe", "Rosalind Achterberg"]


def _money(rng, lo, hi, step):
    return "${:,}".format(rng.randrange(lo, hi, step))


def build_corpus(seed=1729):
    """Returns (chunks, questions).

    chunks: list of dicts {id, doc_id, doc_title, section, text}
    questions: list of dicts {qid, question, answer, gold_chunk_id}
    """
    rng = random.Random(seed)
    chunks, questions = [], []
    cid = 0

    # --- credit agreements -------------------------------------------------
    for i, ent in enumerate(BORROWERS):
        agent = AGENTS[i % len(AGENTS)]
        doc_id = f"CA-{i:03d}"
        doc_title = f"Credit Agreement among {ent}, the Lenders party thereto, and {agent}"
        vals = {
            "fccr": f"{rng.uniform(1.05, 1.60):.2f}",
            "leverage": f"{rng.uniform(3.00, 5.50):.2f}",
            "commfee": f"{rng.uniform(0.15, 0.50):.3f}",
            "debtcap": _money(rng, 5_000_000, 60_000_000, 2_500_000),
            "qdays": str(rng.choice([40, 45, 50, 55, 60])),
            "adays": str(rng.choice([90, 100, 105, 120])),
            "assignmin": _money(rng, 1_000_000, 10_000_000, 500_000),
            "assignfee": _money(rng, 2_500, 5_000, 500),
            "graceper": str(rng.choice([3, 4, 5, 7, 10])),
            "crossdef": _money(rng, 10_000_000, 50_000_000, 5_000_000),
            "margin": str(rng.randrange(110, 320, 5)),
            "abrmargin": str(rng.randrange(10, 220, 5)),
            "default_add": str(rng.choice([2, 3])),
        }
        for sec_title, body, factkey, qtmpl in SECTION_TEMPLATES:
            text = body.format(**vals)
            chunks.append(dict(id=cid, doc_id=doc_id, doc_title=doc_title,
                               section=sec_title, text=text))
            questions.append(dict(qid=len(questions),
                                  question=qtmpl.format(ent=ent),
                                  answer=vals[factkey],
                                  gold_chunk_id=cid,
                                  family="credit"))
            cid += 1

    # --- law firm engagement letters --------------------------------------
    for i, firm in enumerate(FIRMS):
        for j in range(2):
            ent = BORROWERS[(i * 2 + j) % len(BORROWERS)]
            doc_id = f"EL-{i:02d}{j}"
            doc_title = f"Engagement Letter from {firm} to {ent}"
            vals = {
                "firm": firm, "ent": ent,
                "partner": PARTNERS[(i * 2 + j) % len(PARTNERS)],
                "prate": _money(rng, 700, 1600, 25),
                "arate": _money(rng, 350, 800, 25),
                "pararate": _money(rng, 150, 350, 10),
                "payterm": str(rng.choice([15, 30, 45])),
                "retainer": _money(rng, 25_000, 250_000, 5_000),
                "replenish": _money(rng, 10_000, 50_000, 5_000),
                "retention": str(rng.choice([5, 6, 7, 8, 10])),
            }
            for sec_title, body, factkey, qtmpl in ENGAGEMENT_TEMPLATES:
                text = body.format(**vals)
                chunks.append(dict(id=cid, doc_id=doc_id, doc_title=doc_title,
                                   section=sec_title, text=text))
                questions.append(dict(qid=len(questions),
                                      question=qtmpl.format(ent=ent, firm=firm),
                                      answer=vals[factkey],
                                      gold_chunk_id=cid,
                                      family="legal"))
                cid += 1

    return chunks, questions


if __name__ == "__main__":
    c, q = build_corpus()
    print(f"{len(c)} chunks, {len(q)} questions")
    print(f"mean chunk chars: {sum(len(x['text']) for x in c)/len(c):.0f}")
    print("\n--- sample chunk ---")
    print(c[0]["doc_title"], "|", c[0]["section"])
    print(c[0]["text"][:300])
    print("\n--- sample question ---")
    print(q[0]["question"], "->", q[0]["answer"])
