import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { apiService, RecommendRequest, QuestionStats, PredictRequest } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2, Target, Info, AlertCircle } from 'lucide-react';

const MainTab: React.FC = () => {
    const [userId, setUserId] = useState<string>('');
    const [benchmarkTags, setBenchmarkTags] = useState<string>('');
    const [numRecommend, setNumRecommend] = useState<string>('10');
    const [recommendations, setRecommendations] = useState<number[]>([]);
    const [questionStats, setQuestionStats] = useState<QuestionStats[]>([]);
    const [inputRtime, setInputRtime] = useState<string>('');
    const [inputCat, setInputCat] = useState<string>('');
    const [predictions, setPredictions] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);
    const [predictLoading, setPredictLoading] = useState(false);
    const [error, setError] = useState<string>('');
    const [predictError, setPredictError] = useState<string>('');
    const [noResultsMessage, setNoResultsMessage] = useState<string>('');

    // Example valid user IDs for demonstration
    const exampleUserIds = [115, 124, 2746, 5382, 8623, 8701, 12741, 13134, 24418, 24600];

    // 推荐题目
    const handleRecommend = async () => {
        try {
            setLoading(true);
            setError('');
            setNoResultsMessage('');
            setRecommendations([]);
            setQuestionStats([]);
            setPredictions([]);
            setInputRtime('');
            setInputCat('');

            const userIdNum = parseInt(userId.trim());
            const numRecommendNum = parseInt(numRecommend.trim());

            if (isNaN(userIdNum)) {
                throw new Error('Please enter a valid user ID');
            }
            if (isNaN(numRecommendNum) || numRecommendNum <= 0) {
                throw new Error('Please enter a valid number of recommendations');
            }

            const tags = benchmarkTags.trim()
                ? benchmarkTags.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
                : undefined;

            const request: RecommendRequest = {
                user_id: userIdNum,
                benchmark_tags: tags,
                num_recommend: numRecommendNum,
            };

            const response = await apiService.recommendAdvanced(request);
            if (response.question_ids.length === 0) {
                setNoResultsMessage('no_results');
                return;
            }
            setRecommendations(response.question_ids);

            // 获取题目信息
            const statsResponse = await apiService.getQuestionStatsByIds(response.question_ids);
            setQuestionStats(statsResponse);

            // 自动填写类别
            const cats = statsResponse.map(q => {
                return q.part ?? '0';
            });
            setInputCat(cats.join(','));
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Recommendation failed');
        } finally {
            setLoading(false);
        }
    };

    // 预测概率
    const handlePredict = async () => {
        try {
            setPredictLoading(true);
            setPredictError('');
            setPredictions([]);

            // Parse input strings to arrays
            const ids = recommendations;
            const rtime = inputRtime.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const cat = inputCat.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

            if (ids.length === 0 || rtime.length === 0 || cat.length === 0) {
                throw new Error('Please fill in all required inputs');
            }
            if (ids.length !== rtime.length || ids.length !== cat.length) {
                throw new Error('The lengths of question IDs, response times, and categories must match');
            }

            const request: PredictRequest = {
                input_ids: ids,
                input_rtime: rtime,
                input_cat: cat,
            };
            const response = await apiService.predict(request);
            setPredictions(response.probs.slice(0, ids.length));
        } catch (err) {
            setPredictError(err instanceof Error ? err.message : 'Prediction failed');
        } finally {
            setPredictLoading(false);
        }
    };

    const fillExampleUserId = (id: number) => {
        setUserId(id.toString());
        setError('');
        setNoResultsMessage('');
    };

    const chartData = predictions.map((prob, index) => ({
        index: index + 1,
        probability: prob,
    }));

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>Question Recommendation & Prediction</CardTitle>
                    <CardDescription>
                        After recommending questions, question IDs and categories are auto-filled. Please manually fill in response times. Prediction probabilities will be shown automatically.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Info section with example user IDs */}
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center space-x-2 mb-2">
                            <Info className="h-4 w-4 text-blue-600" />
                            <span className="text-sm font-medium text-blue-800">Info</span>
                        </div>
                        <p className="text-sm text-blue-700 mb-2">
                            Please use a user ID that exists in the dataset. Here are some example user IDs:
                        </p>
                        <div className="flex flex-wrap gap-1">
                            {exampleUserIds.slice(0, 8).map((id) => (
                                <button
                                    key={id}
                                    onClick={() => fillExampleUserId(id)}
                                    className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-800 rounded transition-colors"
                                >
                                    {id}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="user-id">User ID</Label>
                        <Input
                            id="user-id"
                            placeholder="e.g. 115"
                            value={userId}
                            onChange={(e) => setUserId(e.target.value)}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="benchmark-tags">Benchmark Tags (optional, comma-separated)</Label>
                        <Input
                            id="benchmark-tags"
                            placeholder="e.g. 1,2,3"
                            value={benchmarkTags}
                            onChange={(e) => setBenchmarkTags(e.target.value)}
                        />
                        <p className="text-sm text-muted-foreground">
                            Benchmark tags: Provided by the dataset, these are one or more detailed tag codes for clustering questions. The meaning of the tags is not provided, but these codes are sufficient to group similar questions together.
                        </p>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="num-recommend">Number of Recommendations</Label>
                        <Input
                            id="num-recommend"
                            placeholder="e.g. 10"
                            value={numRecommend}
                            onChange={(e) => setNumRecommend(e.target.value)}
                        />
                    </div>

                    {error && (
                        <div className="text-red-500 text-sm">{error}</div>
                    )}

                    <div className="flex gap-2">
                        <Button
                            onClick={handleRecommend}
                            disabled={loading}
                            className="flex-1"
                        >
                            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            <Target className="mr-2 h-4 w-4" />
                            Recommend Questions
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Recommendation Results */}
            {recommendations.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>Recommendation Results</CardTitle>
                        <CardDescription>
                            List of questions recommended based on user history and benchmark tags
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {/* Recommended Question Info */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                            {recommendations.map((qid, index) => {
                                const stat = questionStats.find(q => q.question_id === qid);
                                return (
                                    <div key={index} className="text-left p-2 bg-muted rounded border">
                                        <div className="text-sm text-muted-foreground mb-1">Recommendation {index + 1}</div>
                                        <div className="font-semibold mb-1">Question {qid}</div>
                                        <div className="flex flex-wrap gap-1 mb-1">
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-orange-100 text-orange-800">
                                                Difficulty: {stat?.difficulty !== undefined && stat?.difficulty !== null ? stat.difficulty.toFixed(3) : '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-green-100 text-green-800">
                                                Discrimination: {stat?.discrimination !== undefined && stat?.discrimination !== null ? stat.discrimination.toFixed(3) : '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-yellow-100 text-yellow-800">
                                                Tags: {stat?.tags ?? '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-pink-100 text-pink-800">
                                                TOEIC Part: {stat?.part ?? '-'}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Prediction Input */}
            {recommendations.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>Prediction Input</CardTitle>
                        <CardDescription>
                            Question IDs and categories are auto-filled. Please enter response times (comma-separated, must match the number of questions)
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="input-ids">Question ID Sequence</Label>
                            <Input
                                id="input-ids"
                                value={recommendations.join(',')}
                                readOnly
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="input-cat">Category Sequence</Label>
                            <Input
                                id="input-cat"
                                value={inputCat}
                                readOnly
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="input-rtime">Response Time Sequence (comma-separated)</Label>
                            <Input
                                id="input-rtime"
                                placeholder={`e.g. ${Array(recommendations.length).fill(1000).join(',')}`}
                                value={inputRtime}
                                onChange={(e) => setInputRtime(e.target.value)}
                            />
                        </div>
                        {predictError && (
                            <div className="text-red-500 text-sm">{predictError}</div>
                        )}
                        <Button onClick={handlePredict} disabled={predictLoading} className="w-full">
                            {predictLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            {predictLoading ? 'Predicting...' : 'Start Prediction'}
                        </Button>
                    </CardContent>
                </Card>
            )}

            {/* Prediction Results */}
            {predictions.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>Prediction Results</CardTitle>
                        <CardDescription>
                            Model predicted probability for each question
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="index" />
                                    <YAxis domain={[0, 1]} />
                                    <Tooltip
                                        formatter={(value: number) => [value.toFixed(4), 'Predicted Probability']}
                                        labelFormatter={(label) => `Question ${label}`}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="probability"
                                        stroke="#8884d8"
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
                            {predictions.map((prob, index) => (
                                <div key={index} className="text-center p-2 bg-muted rounded">
                                    <div className="text-sm text-muted-foreground">Question {index + 1}</div>
                                    <div className="font-semibold">{prob.toFixed(4)}</div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            {noResultsMessage && (
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                            <AlertCircle className="h-5 w-5 text-orange-500" />
                            <span>No Recommendation Results Found</span>
                        </CardTitle>
                        <CardDescription>
                            No recommendation results found for user ID {userId}
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                            <p className="text-sm text-orange-800 mb-2">Possible reasons:</p>
                            <ul className="text-sm text-orange-700 space-y-1 ml-4">
                                <li>• The user ID does not exist in the dataset</li>
                                <li>• The user does not have enough historical data</li>
                                <li>• The user has already attempted all questions</li>
                            </ul>
                        </div>
                        <div>
                            <p className="text-sm text-muted-foreground mb-2">Please try the following example user IDs:</p>
                            <div className="flex flex-wrap gap-2">
                                {exampleUserIds.slice(0, 5).map((id) => (
                                    <Button
                                        key={id}
                                        onClick={() => fillExampleUserId(id)}
                                        variant="outline"
                                        size="sm"
                                    >
                                        User {id}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
};

export default MainTab; 