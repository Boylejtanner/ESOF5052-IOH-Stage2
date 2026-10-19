# Stage 2 Submission and GitHub Access

## Deliverables

The Stage 2 requirements are developed code with meaningful documentation, an expanded report with an introduction and literature review, and a GitHub or Colab link. This repository provides the code, reproduction guidance, selected aggregate evidence, and the LaTeX report. The author compiles and submits the PDF separately.

Submit:
1. The PDF compiled from `report/main.tex`.
2. The GitHub repository URL.
3. If useful, the submitted commit hash or a release/tag name so the grading version is unambiguous.

The report includes 28 bibliography entries, including more than ten peer-reviewed sources. The Stage 2 status is documented in the root README. The existing BSHI training trials use the implemented input recipes; the newly selected context has not yet been trained.

## Publish the local repository with GitHub Desktop

1. Open GitHub Desktop.
2. Select **File > Options > Accounts > Sign in to GitHub.com**. Complete the browser sign-in and two-factor authentication yourself. Do not put passwords, tokens, or recovery codes in the repository or a chat.
3. Select **File > Add local repository** and choose the `ESOF5052-IOH-Stage2` folder.
4. Review the existing initial commit. The intended contents are code, notebooks without outputs, configuration, selected aggregate results, documentation, and `report/main.tex`.
5. Select **Publish repository**. Use a name such as `ESOF5052-IOH-Stage2` and keep **Keep this code private** checked.
6. Confirm the correct GitHub account or organization, then publish.
7. Select **Repository > View on GitHub**. Copy the displayed repository URL for the submission.

A private repository and an anonymous-access link are different access models. GitHub does not provide an unlisted, anyone-with-the-link mode for a normal private repository.

## Give graders access while keeping the repository private

1. Obtain the instructor's and TA's GitHub usernames or the course's required access mechanism.
2. On the repository website, open **Settings > Collaborators** (or **Collaborators and teams** for an organization).
3. Select **Add people** and invite the grader accounts. Use read-only access if the account type and organization settings provide it.
4. Ask the graders to accept the invitations before grading. They must be signed in to an invited GitHub account when opening the link.
5. Submit the repository URL and note that access was granted to those accounts.

On a personal repository, collaborator permissions can be broader than read-only; consult the permission display before inviting. Do not share your own account credentials.

## If the course explicitly requires anyone with the link to open it

That requires a **public** repository. Change visibility only after checking course policy and data-provider terms, and only for this curated code repository. Never make the research workspace or patient data public. Public visibility means anyone can discover and clone the repository, not merely people who receive the link.

The default for this submission is private. Do not change it to public solely to avoid inviting graders.

## Final checklist

- [ ] The correct `.tex` source compiles and the PDF is attached to the course submission.
- [ ] The repository contains the documented Stage 2 commit.
- [ ] Raw data, identifiers, model weights, notebook outputs, credentials, and local logs are absent.
- [ ] The link opens for the invited grader account, not just the author's account.
- [ ] The README clearly distinguishes completed trials from the selected model's next integration stage.
- [ ] The final PDF and repository link are submitted before the course deadline.
