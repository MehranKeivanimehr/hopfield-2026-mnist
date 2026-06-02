$ClusterHost = "m.keivanimehr@ece-focus-xg01.ad.ufl.edu"
$RemoteRoot = "~/research/cmkc/project"
$LocalRoot = "C:\Users\m.keivanimehr\OneDrive - University of Florida\Documents\New project"

ssh $ClusterHost "mkdir -p $RemoteRoot"
scp -r `
  "$LocalRoot\README.md" `
  "$LocalRoot\proposal.md" `
  "$LocalRoot\references.bib" `
  "$LocalRoot\pyproject.toml" `
  "$LocalRoot\train.py" `
  "$LocalRoot\evaluate.py" `
  "$LocalRoot\configs" `
  "$LocalRoot\docs" `
  "$LocalRoot\scripts" `
  "$LocalRoot\src" `
  $ClusterHost":"$RemoteRoot
