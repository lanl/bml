# Check basic commit message formatting
#
# vim: syntax=ruby

# Lint the commit message.
commit_lint.check fail: :all

# Ensure a clean commit history.
if git.commits.any? { |c| c.message =~ /^Merge branch/ }
  fail('Please rebase to get rid of the merge commits in this PR')
end

# Look for prose issues.
# prose.lint_files

# Running plugin with reviewers count specified. Find maximum two
# reviewers.
mention.run(2, [], [])

# Show the greeting.
welcome_message.greet

# Local Variables:
# mode: ruby
# End:
