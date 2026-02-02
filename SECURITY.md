# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in SunBURST, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly (see GitHub profile for contact information)
3. Include a description of the vulnerability, steps to reproduce, and potential impact

You should receive an acknowledgment within 48 hours. We will work with you to understand and address the issue before any public disclosure.

## Scope

SunBURST is a scientific computing library. Its primary security considerations are:

- **Arbitrary code execution:** SunBURST evaluates user-provided likelihood functions. Users should only run trusted code, as with any scientific computing library.
- **GPU memory:** Large-dimension problems allocate significant GPU memory. The library includes safeguards against out-of-memory conditions but users should monitor resource usage.
- **Dependencies:** SunBURST depends on NumPy, SciPy, and optionally CuPy. Keep these packages updated to their latest security-patched versions.

## Best Practices

- Install SunBURST from PyPI (`pip install sunburst-bayes`) or directly from the GitHub repository
- Verify package integrity via PyPI checksums
- Do not execute untrusted likelihood functions
