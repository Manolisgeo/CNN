



transform = transforms.Compose([
    transforms.ToTensor(),             # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ----------------------------
# 3. Initialize model, loss, optimizer
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 4. Training loop
# ----------------------------
num_epochs = 5

for epoch in range(1, num_epochs + 1):
    model.train()  # set model to training mode
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # reset gradients
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                # backprop
        optimizer.step()               # update parameters

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch}, Batch {batch_idx + 1}] Loss: {running_loss / (batch_idx + 1):.4f}")
            import sys; sys.stdout.flush()
        if batch_idx % 100 == 99:
            print(f"[Epoch {epoch}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # ----------------------------
    # 5. Evaluate on test set
    # ----------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch} Test Accuracy: {100 * correct / total:.2f}%\n")

# ----------------------------
# 6. Save the trained model
# ----------------------------
torch.save(model.state_dict(), "simple_cnn_mnist.pth")
print("Model saved as simple_cnn_mnist.pth")

