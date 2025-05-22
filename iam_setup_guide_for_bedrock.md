# AWS IAM Setup for Bedrock Access

This guide will walk you through creating an IAM user with the necessary permissions to access AWS Bedrock services.

## Step 1: Create an IAM Policy for Bedrock

1. Sign in to the AWS Management Console and open the IAM console at https://console.aws.amazon.com/iam/
2. In the navigation pane, choose **Policies** and then click **Create policy**
3. Select the **JSON** tab and paste the following policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel",
                "bedrock:ListCustomModels",
                "bedrock:GetCustomModel",
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "*"
        }
    ]
}
```

4. Click **Next: Tags** (add tags if desired)
5. Click **Next: Review**
6. Name the policy `BedrockAccessPolicy` and provide a description
7. Click **Create policy**

## Step 2: Create an IAM User

1. In the IAM console navigation pane, choose **Users** and then click **Add users**
2. Enter a user name (e.g., `bedrock-api-user`)
3. Select **Access key - Programmatic access** as the access type
4. Click **Next: Permissions**
5. Choose **Attach existing policies directly**
6. Search for and select the `BedrockAccessPolicy` you created
7. Click **Next: Tags** (add tags if desired)
8. Click **Next: Review** and verify the details
9. Click **Create user**

## Step 3: Secure your Access Keys

1. On the success page, you'll see the **Access key ID** and **Secret access key**
2. Click **Download .csv** to save these credentials securely
3. **IMPORTANT**: This is your only opportunity to view or download the secret access key

## Step 4: Enable Model Access in Bedrock

1. Go to the Amazon Bedrock console at https://console.aws.amazon.com/bedrock/
2. Navigate to **Model access** in the left sidebar
3. Click **Manage model access**
4. Select the models you want to use (e.g., Claude 3 Sonnet, Haiku)
5. Click **Request model access**
6. Once approved (usually immediate), your IAM user can access these models

## Step 5: Update your .env file

Add your credentials to your `.env` file:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1  # Change to your preferred region
```

## Security Best Practices

1. **Never** commit your `.env` file to version control
2. Consider using IAM roles instead of access keys for production environments
3. Regularly rotate your access keys
4. Monitor your AWS usage and set up billing alerts to prevent unexpected charges
5. Apply the principle of least privilege by refining the IAM policy to only include necessary permissions
